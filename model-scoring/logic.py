import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, log_loss
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


from plotnine import (
    ggplot, aes, geom_col, geom_text, coord_flip, theme_minimal,
    labs, theme, element_text
)

def detect_separator(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        return ';' if first_line.count(';') > first_line.count(',') else ','

def load_and_process(file_path: str):
    sep = detect_separator(file_path)
    df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
    df = df.replace(',', '.', regex=True)

    if 'Class' in df.columns:
        y_col = 'Class'
    elif 'Target' in df.columns:
        y_col = 'Target'
    else:
        raise ValueError("Brak kolumny 'Class' lub 'Target' w danych wejściowych.")

    X = df.drop(columns=[y_col]).astype(float)
    y = df[y_col]

    X = X.fillna(X.mean())

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = y.map({'A': 0, 'C': 1, 'B': 1 , 'G': 0 , 'O' : 1, 'OA': 0 , 'ON': 1})

    noise = np.random.uniform(0.001, 0.15, size=X.shape)
    X['random_noise'] = noise.mean(axis=1)

    return X, y.values, df

def set_params(learning_rate = 0.1, col_sample = 0.5):

    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'colsample_bytree': col_sample,
        'colsample_bylevel': col_sample, 
        'colsample_bynode': col_sample,
        'subsample': 0.7,
        'learning_rate': learning_rate,
        'seed': 42
    }
     
    return param
    

def metric_xgboost(X,y,best_params, num_round):
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "logloss": [],
        "roc_auc_score": []
    }

    models = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(best_params, dtrain, num_round)
        models.append(bst)

        y_pred_proba_test = bst.predict(dtest)
        y_pred_test = [1 if prob >= 0.5 else 0 for prob in y_pred_proba_test]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred_test))
        metrics["precision"].append(precision_score(y_test, y_pred_test))
        metrics["recall"].append(recall_score(y_test, y_pred_test))
        metrics["f1"].append(f1_score(y_test, y_pred_test))
        metrics["logloss"].append(log_loss(y_test, y_pred_proba_test))
        metrics["roc_auc_score"].append(roc_auc_score(y_test, y_pred_proba_test))

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

    avg_metrics["model_bst"] = models[-1]
    return avg_metrics

def metric_xgboost_selection(X,y, bst, best_params, num_round, treshold):
    importance = bst.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Cecha': list(importance.keys()),
        'Istotność': list(importance.values())
    })

    # pobranie wartosci szumu
    noise_importance = importance.get('random_noise', 0)

    # pobranie wartosci mean/treshold/median
    thresholds = {
        'noise': noise_importance,
        'mean': importance_df['Istotność'].mean(),
        'threshold': importance_df['Istotność'].quantile(treshold),
        'median': importance_df['Istotność'].median()
    }

    selected_features = {
        name: importance_df[importance_df['Istotność'] > value]['Cecha']
        for name, value in thresholds.items()
    }

    data_selection_xgboost = {
        name: metric_xgboost(X[features], y, best_params, num_round)
        for name, features in selected_features.items()
    }
    return data_selection_xgboost


def xgb_test_selection(X, y, bst, param, num_round, treshold):
    importance = bst.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Cecha': list(importance.keys()),
        'Istotność': list(importance.values())
    })

    # pobranie wartosci szumu
    noise_importance = importance.get('random_noise', 0)

    # pobranie wartosci mean/treshold/median
    mean_importance = importance_df['Istotność'].mean()
    threshold = importance_df['Istotność'].quantile(treshold)
    median_importance = importance_df['Istotność'].median()
    selected_features_noise = importance_df[importance_df['Istotność'] > noise_importance]['Cecha']
    selected_features_mean = importance_df[importance_df['Istotność'] > mean_importance]['Cecha']
    selected_features_treshold = importance_df[importance_df['Istotność'] > threshold]['Cecha']
    selected_features_median = importance_df[importance_df['Istotność'] > median_importance]['Cecha']

    # funkcja pomocnicza 
    def run_cv_auc(X_selected, label=y):
        dtrain_sub = xgb.DMatrix(X_selected, label=label)
        results = xgb.cv(
                param,
                dtrain_sub,
                num_boost_round=num_round,
                nfold=5,
                stratified=True,
                shuffle=True,
                early_stopping_rounds=50,
                seed=42,
                metrics='auc',
                verbose_eval=False
            )
        return results['test-auc-mean'].iloc[-1]

    # AUCy 
    auc_all = run_cv_auc(X[importance_df['Cecha']])
    auc_mean = run_cv_auc(X[selected_features_mean])
    auc_threshold = run_cv_auc(X[selected_features_treshold])
    auc_median = run_cv_auc(X[selected_features_median])
    auc_noise =run_cv_auc(X[selected_features_noise])

    # print(f"- AUC (wszystkie cechy):            {auc_all:.4f}")
    # print(f"- AUC (powyżej średniej):           {auc_mean:.4f}")
    # print(f"- AUC (powyżej threshold {treshold}):     {auc_threshold:.4f}")
    # print(f"- AUC (powyżej mediany):            {auc_median:.4f}")

    return {
        "auc_all": auc_all,
        "auc_mean": auc_mean,
        "auc_threshold": auc_threshold,
        "auc_median": auc_median,
        "auc_noise": auc_noise,
    }

def generate_importance_plot(bst, importance_type='gain'):
    importance_gain = bst.get_score(importance_type='gain')
    importance_cover = bst.get_score(importance_type='cover')
    importance_weight = bst.get_score(importance_type='weight')

    importance_df_score = pd.DataFrame({
        'Cecha': list(importance_gain.keys()),  
        'Gain': list(importance_gain.values()),  
        'Cover': [importance_cover.get(feat, 0) for feat in importance_gain.keys()], # dodanie 0, jezeli brak cechy 
        'Weight': [importance_weight.get(feat, 0) for feat in importance_gain.keys()]  # dodanie 0, jezeli brak cechy
    })

    if importance_type == "gain":
        importance_dict = bst.get_score(importance_type='gain')
        importances_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
        importances_df.index.name = 'feature'
        importances_df.reset_index(inplace=True)
        importances_df['gain_percent'] = 100 * importances_df['gain'] / importances_df['gain'].sum()

        features_number = 20
        top_features = importances_df.sort_values(by='gain', ascending=False).head(features_number).copy()
        top_features['gain_percent_label'] = top_features['gain_percent'].map(lambda x: f"{x:.1f}%")

        top_features['feature'] = pd.Categorical(
            top_features['feature'],
            categories=top_features['feature'][::-1],
            ordered=True
        )

        p = (
            ggplot(top_features, aes(x='feature', y='gain'))
            + geom_col(fill='skyblue')
            + geom_text(
                aes(label='gain_percent_label'),
                ha='left', nudge_y=0.02, size=8
            )
            + coord_flip()
            + theme_minimal()
            + labs(
                title=f'Ważność cech wg gain (suma: {importances_df["gain"].sum():.2f})',
                x='Cechy',
                y='Gain'
            )
            + theme(
                axis_text_y=element_text(size=10),
                axis_title=element_text(size=12),
                plot_title=element_text(size=14)
            )
        )
    else:
        importance_df_score['Combined_score'] = (importance_df_score['Weight'] + importance_df_score['Cover']) * importance_df_score['Gain']

        # Posortuj malejąco po Combined_score i wybierz top 20
        top_features = importance_df_score.sort_values(by='Combined_score', ascending=False).head(20).copy()

        # Oblicz procentowy udział
        top_features['combined_percent_label'] = top_features['Combined_score'].map(lambda x: f"{100*x/top_features['Combined_score'].sum():.1f}%")

        top_features['Cecha'] = pd.Categorical(
            top_features['Cecha'],
            categories=top_features['Cecha'][::-1],
            ordered=True
        )

        p = (
            ggplot(top_features, aes(x='Cecha', y='Combined_score'))
            + geom_col(fill='skyblue')
            + geom_text(
                aes(label='combined_percent_label'),
                ha='left', nudge_y=0.02, size=8
            )
            + coord_flip()
            + theme_minimal()
            + labs(
                title=f'Ważność cech wg combined score (suma: {top_features["Combined_score"].sum():.2f})',
                x='Cechy',
                y='Combined Score'
            )
            + theme(
                axis_text_y=element_text(size=10),
                axis_title=element_text(size=12),
                plot_title=element_text(size=14)
            )
        )


    return p, importance_df_score


def get_number_of_correct(df, importance_df_score, importance_type):
    columns_with_irr_or_noise = [col for col in df.columns if 'Irr' in col or 'Noise' in col or "Class" in col or "Target" in col]
    num_columns_with_irr_or_noise = len(columns_with_irr_or_noise)

    columns_without_irr_or_noise = [col for col in df.columns if col not in columns_with_irr_or_noise]
    num_columns_without_irr_or_noise = len(columns_without_irr_or_noise)
    if importance_type == "gain":
        importance_df_sorted_results_gain = importance_df_score.sort_values(by='Gain', ascending=False)
        importance_df_sorted_results_gain['Is_Importance'] = ~importance_df_sorted_results_gain['Cecha'].str.contains("Noise|Irr", regex=True)
        top_x_gain = importance_df_sorted_results_gain.head(num_columns_without_irr_or_noise)
        gain_score = top_x_gain['Is_Importance'].sum()
    else:
        importance_df_sorted_results = importance_df_score.sort_values(by='Combined_score', ascending=False)
        importance_df_sorted_results['Is_Importance'] = ~importance_df_sorted_results['Cecha'].str.contains("Noise|Irr", regex=True)
        top_x = importance_df_sorted_results.head(num_columns_without_irr_or_noise)
        combined_score = top_x['Is_Importance'].sum()

    if importance_type == "gain":
        top_selected = gain_score
    else:
        top_selected = combined_score

    xgboost_finded_importance = top_selected;

    return xgboost_finded_importance, num_columns_without_irr_or_noise


def boruta_features(X,y, best_params, num_round, learning_rate, col_sample, max_boruta, min_boruta):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=num_round,  # liczba drzew
        max_depth=None,  # glebokosc im nizsza wartosc tym wieksza mozliwosc przeuczenia w malych danych
        min_samples_split=5,  # dzielenie wezla tylko jezeli x obserwacji
        class_weight='balanced',  # w przypadku gdybysmy mieli o wiele wiecej wartosc 0 w class niz 1 to zostana rownomieniernie rozlozone dzieki balanced
        random_state=42
    )

    boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
    boruta.fit(X_train.values, y_train)

    # tworzenie listy cech i ich rang
    features = X.columns
    ranks = boruta.ranking_
    selected = boruta.support_

    # sortowanie cech według rangi
    sorted_indices = np.argsort(ranks)
    features_sorted = features[sorted_indices]
    ranks_sorted = ranks[sorted_indices]

    boruta_top_features = X.columns[np.where((ranks >= max_boruta) & (ranks <= min_boruta))[0]]

    X_selected = X[boruta_top_features]

    top_n = len(boruta_top_features) + 3

    xgb_metrics_with_boruta_features =  metric_xgboost(X_selected, y, best_params, num_round)

    #xgb_no_use, xgb_metrics_corss_boruta_features, best_params_no_used, num_rounds_no_used = train_xgboost(X,y, learning_rate, num_round, col_sample)
    features_top = features_sorted[:top_n]
    ranks_top = ranks_sorted[:top_n]
    selected_top = [selected[np.where(features == f)[0][0]] for f in features_top]

    colors_top = ['red' if sel else 'green' for sel in selected_top]


    top_features_df = pd.DataFrame({
        'Cecha': features_top,
        'Ranga': ranks_top,
        'Kolor': colors_top
    })

    top_features_df['Cecha'] = pd.Categorical(
        top_features_df['Cecha'],
        categories=top_features_df['Cecha'][::-1],
        ordered=True
    )

    top_features_df = top_features_df.sort_values(by='Ranga', ascending=True).copy()

    top_features_df['ranga_label'] = top_features_df['Ranga'].astype(str)


    p = (
        ggplot(top_features_df, aes(x='Cecha', y='Ranga', fill='Kolor'))
        + geom_col(show_legend=False)
        + coord_flip()
        + theme_minimal()
        + labs(
            title=f'Najważniejsze cechy według metody Boruta (top {top_n})',
            x='Cechy',
            y=f'Ranga cechy (niższa = ważniejsza), liczba cech z rangą jeden: {len(boruta_top_features)}'
        )
        + theme(
            axis_text_y=element_text(size=6),   
            axis_title=element_text(size=10),
            plot_title=element_text(size=12),
            figure_size=(25, 50) 
        )
    )
    
    return p, xgb_metrics_with_boruta_features


def generate_comparison_plot(data_xgb_selection_all, xgb_data_with_boruta, data_xgb_metrics):

    best_key = max(data_xgb_selection_all, key=lambda k: data_xgb_selection_all[k]['accuracy'])
    best_selection = data_xgb_selection_all[best_key]

    metrics = ["Accuracy", "F1", "AUC"]
    values = [
        [best_selection["accuracy"], best_selection["f1"], best_selection["roc_auc_score"]],
        [xgb_data_with_boruta["accuracy"], xgb_data_with_boruta["f1"], xgb_data_with_boruta["roc_auc_score"]],
        [data_xgb_metrics["accuracy"], data_xgb_metrics["f1"], data_xgb_metrics["roc_auc_score"]]
    ]

    df_plot = pd.DataFrame({
        "Metryka": metrics * 3,
        "Wartość": [round(v, 5) for row in values for v in row],
        "Model": (["XGBoost – najlepsze cechy"] * 3 
                  + ["XGBoost – Boruta"] * 3 
                  + ["XGBoost – pełny model"] * 3)
    })

    p = (
        ggplot(df_plot, aes(x="Metryka", y="Wartość", fill="Model"))
        + geom_col(position="dodge")
        + labs(
            title="Porównanie metryk modeli XGBoost",
            x="Metryka",
            y="Wartość"
        )
        + theme_minimal()
        + theme(
            axis_text=element_text(size=10),
            axis_title=element_text(size=12),
            plot_title=element_text(size=14),
            legend_position="bottom"
        )
    )
    return p





def results_data(file_path: str, n_estimators , learning_rate, col_sample, importance_type, treshold, max_boruta , min_boruta):
    # ladowanie danych
    X, y, df = load_and_process(file_path)
    print("Dane załadowano")
    # wyniki xgboost
    best_params = set_params(learning_rate, col_sample)

    print("Model, AUC, Parametry - Wyznacozne")
    data_xgb = metric_xgboost(X, y, best_params, n_estimators)
    print("Metryki Wyznaczone")
    # data_xgb_selection = xgb_test_selection(X, y, data_xgb['model_bst'], best_params, n_estimators, treshold)
    # print("Metryki Selekcjonowane Wyznaczone AUC")
    data_xgb_selection_all =  metric_xgboost_selection(X,y, data_xgb['model_bst'], best_params, n_estimators, treshold)
    # wykresy xgboost
    gain_plot_xgb, importance_df_score = generate_importance_plot(data_xgb['model_bst'], importance_type)
    print("Wykresy wygenerowane") 
    # sprawdzanie dzialana na syntetycznych
    finded_importance_xgb, number_of_importance = get_number_of_correct(df, importance_df_score, importance_type)
    #boruta
    plot_boruta, xgb_data_with_boruta = boruta_features(X,y, best_params, n_estimators, learning_rate, col_sample, max_boruta, min_boruta)
    #porownanie
    comparison_plot = generate_comparison_plot(data_xgb_selection_all, xgb_data_with_boruta, data_xgb)

    return {
        "x": X,
        "y": y,
        "df": df,
        "xgb": xgb,
        "best_params": best_params,
        "data_xgb_metrics": data_xgb,
        # "data_xgb_selection": data_xgb_selection,
        "gain_plot_xgb": gain_plot_xgb,
        "finded_importance_xgb": finded_importance_xgb,
        "number_of_importance": number_of_importance,
        "data_xgb_selection_all": data_xgb_selection_all,
        "plot_boruta": plot_boruta,
        "xgb_data_with_boruta":xgb_data_with_boruta,
        "comparison_plot":comparison_plot,
    }
