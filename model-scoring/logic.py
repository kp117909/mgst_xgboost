import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = y.map({'A': 0, 'C': 1, 'B': 1})

    noise = np.random.uniform(0.001, 0.15, size=X.shape)
    X['random_noise'] = noise.mean(axis=1)

    return X, y.values, df

def train_xgboost(X,y, learning_rate = 0.1, n_estimators = 100, col_sample = 0.5):
    dtrain = xgb.DMatrix(X, label=y)

    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 2,
        'colsample_bytree': col_sample,
        'colsample_bylevel': col_sample, 
        'colsample_bynode': col_sample,
        'subsample': 0.7,
        'learning_rate': learning_rate,
        'seed': 42
    }


    skf_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=n_estimators,
        nfold=10,
        shuffle=True,
        stratified=True,
        seed=42,
        metrics='auc',
        early_stopping_rounds=100, 
        verbose_eval=False
    )

    auc = skf_results['test-auc-mean'].iloc[-1]
    print('test_auc_mean', skf_results['test-auc-mean'].iloc[-1])
    print('train', skf_results['train-auc-mean'].iloc[-1])
    print('std',  skf_results['test-auc-std'].iloc[-1])

    return  xgb, auc, param
    

def metric_xgboost(X,y,best_params, num_round):
    # przemieszanie danych przed podziałem
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    # podział na dane treningowe i testowe (70% treningowe, 30% testowe)
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.3, random_state=42)

    # konwersja danych do formatu DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # trenowanie modelu
    num_round = num_round
    bst = xgb.train(best_params, dtrain, num_round)

    # predykcje na zbiorze testowym
    y_pred_proba_test = bst.predict(dtest)
    y_pred_test = [1 if prob >= 0.5 else 0 for prob in y_pred_proba_test]

    # policzenie odpowiednich miar 
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    logloss_test = log_loss(y_test, y_pred_proba_test)

    return {
        "accuracy": accuracy_test,
        "precision": precision_test,
        "recall": recall_test,
        "f1": f1_test,
        "logloss": logloss_test,
        "model_bst": bst,
    }

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
                seed=42,
                metrics='auc',
                early_stopping_rounds=20, # zatrzymuje trening wczesniej jezeli model nie poprawia sie w przez określoną liczbe rund
                verbose_eval=False
            )
        return results['test-auc-mean'].iloc[-1]

    # AUCy 
    auc_all = run_cv_auc(X[importance_df['Cecha']])
    auc_mean = run_cv_auc(X[selected_features_mean])
    auc_threshold = run_cv_auc(X[selected_features_treshold])
    auc_median = run_cv_auc(X[selected_features_median])
    auc_noise =run_cv_auc(X[selected_features_noise])

    print(f"- AUC (wszystkie cechy):            {auc_all:.4f}")
    print(f"- AUC (powyżej średniej):           {auc_mean:.4f}")
    print(f"- AUC (powyżej threshold 0.9):     {auc_threshold:.4f}")
    print(f"- AUC (powyżej mediany):            {auc_median:.4f}")

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




def results_data(file_path: str, n_estimators , learning_rate, col_sample, importance_type, treshold):
    # ladowanie danych
    X, y, df = load_and_process(file_path)
    print("Dane załadowano")
    # wyniki xgboost
    xgb, best_auc, best_params = train_xgboost(X,y, learning_rate, n_estimators, col_sample)
    print("Model, AUC, Parametry - Wyznacozne")
    data_xgb = metric_xgboost(X, y, best_params, n_estimators)
    print("Metryki Wyznaczone")
    data_xgb_selection = xgb_test_selection(X, y, data_xgb['model_bst'], best_params, n_estimators, treshold)
    print("Metryki Selekcjonowane Wyznaczone AUC")
    data_xgb_selection_all =  metric_xgboost_selection(X,y, data_xgb['model_bst'], best_params, n_estimators, treshold)
    # wykresy xgboost
    gain_plot_xgb, importance_df_score = generate_importance_plot(data_xgb['model_bst'], importance_type)
    print("Wykresy wygenerowane") 

    # sprawdzanie dzialana na syntetycznych
    finded_importance_xgb, number_of_importance = get_number_of_correct(df, importance_df_score, importance_type)

    return {
        "x": X,
        "y": y,
        "df": df,
        "xgb": xgb,
        "best_auc": best_auc,
        "best_params": best_params,
        "data_xgb_metrics": data_xgb,
        "data_xgb_selection": data_xgb_selection,
        "gain_plot_xgb": gain_plot_xgb,
        "finded_importance_xgb": finded_importance_xgb,
        "number_of_importance": number_of_importance,
        "data_xgb_selection_all": data_xgb_selection_all,

    }
