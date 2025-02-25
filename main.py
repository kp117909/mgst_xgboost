import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

# wczytanie pliku CSV
df = pd.read_csv("surowica_chory_zdrowy.csv", sep=';')
df = df.replace(',', '.', regex=True)

# pobranie wartości
X = df.drop('Class', axis=1)
# preprocessing
X = X.astype(float)
# pobranie predykcji
y = df['Class'].values

# definicja paremetrow dla xgboost
# korzystałem z strony https://xgboost.readthedocs.io/en/stable/python/python_intro.html
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}

# max_depth -> maksymalna głebokosc drzewa, wyzsze wartosci prowadza do overfittingu
# eta -> predkosc uczenia
# konwersja danych DMatrix do formatu XGboost
dtrain = xgb.DMatrix(X, label=y)

# wykonanie kroswalidacji
skf_results = xgb.cv(param, dtrain, num_boost_round=10, nfold=5, stratified=True, seed=42, metrics='auc')
# AUC czyli zdolność do rozrózniania między klasami
print(skf_results)
print(f"Średnia wartość AUC: {skf_results['test-auc-mean'].iloc[-1]}")

# kroswalidacja za pomoca funkcji skf a nie algorytmu xgb dla większej ilosci wyników
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = [] #  procent do poprawnych przewidywań w stosunku do wszystkich przewidywań
precisions = [] #  pokazuje jak wiele przypadkow pozytywnych jest prawdziwie pozytywnych
recalls = [] # pokazuje jak wiele rzeczywiscie pozytywnych przypadkow model poprawnie wykryl
f1_scores = []  # harmoniczna srednia precyzji i czulosci
log_losses = [] # roznica pomiedzy przewidywaniami a rzeczywywistymi klasami im mniej tym lepiej
aucs = [] # zdolnosc do rozrozniania miedzy klasami, lecz za pomoca innej biblioteki
bst_models = []  # lista na modele

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # konwersja danych do formatu DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # trenowanie modelu
    num_round = 10
    bst = xgb.train(param, dtrain, num_round)
    bst_models.append(bst)  # zapisanie modelu

    # predykcja
    y_pred_proba = bst.predict(dtest)
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_proba]

    # obliczanie różnych miar
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    # zbieranie wyników
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    log_losses.append(logloss)
    aucs.append(auc)

# wyświetlenie średnich wyników
print(f"Średnia dokładność: {np.mean(accuracies) * 100:.2f}%")
print(f"Średnia precyzja: {np.mean(precisions) * 100:.2f}%")
print(f"Średnia czułość: {np.mean(recalls) * 100:.2f}%")
print(f"Średni F1 Score: {np.mean(f1_scores) * 100:.2f}%")
print(f"Średni Log Loss: {np.mean(log_losses):.4f}")
print(f"Średni AUC: {np.mean(aucs):.4f}")

# wyświetlenie wykresu istotności cech
xgb.plot_tree(bst_models[0])  # uzycie ostatniego modelu z kroswalidacji
plt.title('Ważność cech')
plt.xlabel('Ważność')
plt.ylabel('Cechy')
plt.show()