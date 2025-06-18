

import pandas as pd
import xgboost as xgb
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def load_and_process_data(file_path: str):
    # Wczytaj dane (dostosuj do formatu w notebooku)
    df = pd.read_csv(file_path)
    X = df.drop(columns=['target'])
    y = df['target']

    # Model XGBoost
    model = xgb.XGBClassifier()
    model.fit(X, y)
    booster = model.get_booster()

    # GAIN z XGBoost
    gain_dict = booster.get_score(importance_type='gain')
    importances_df = pd.DataFrame.from_dict(gain_dict, orient='index', columns=['gain'])
    importances_df.index.name = 'feature'
    importances_df.reset_index(inplace=True)
    importances_df['gain_percent'] = 100 * importances_df['gain'] / importances_df['gain'].sum()

    # Boruta
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
    boruta.fit(X.values, y.values)

    boruta_df = pd.DataFrame({
        'feature': X.columns,
        'boruta_rank': boruta.ranking_
    })

    # zwracamy gotowe dane
    return importances_df, boruta_df
