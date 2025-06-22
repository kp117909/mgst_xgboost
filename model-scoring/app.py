from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from logic import results_data
import pandas as pd
import numpy as np
app_ui = ui.page_navbar(
    ui.nav_spacer(),
     ui.nav_panel(
        "Wyniki modelu",
        ui.navset_card_underline(
            ui.nav_panel(
                "XGB",
               ui.tags.div(
                    ui.h4("Wyniku algorytmu na całym zbiorze"),
                    ui.output_table("xgb_all_metrics_table"),
                    ui.h4("Wyniki algorytmu na zbiorze cech wybranych przez XGBoost"),
                    ui.output_table("xgb_all_metrics_selection_table")
                ),
                # ui.tags.div(
                #     ui.tags.div(
                #         ui.h4("Wyniki algorytmu na zbiorze cech wybranych przez XGBoost - Kroswalidacja :"),
                #         ui.output_table("xgb_auc_selection_table")
                #     )
                # ),
                ui.card(
                    ui.card_header("Wykresy cech istotnych"),
                    ui.input_select("importance_type", "Rodzaj hyperparametru:", {
                        "gain": "Gain",
                        "combined_score": "Combined Score"
                    }),
                    ui.output_image("gain_plot"),
                    # ui.output_ui("importance_summary")
                )
            ),
            ui.nav_panel(
                "Boruta",
                ui.tags.div(
                    ui.tags.div(
                        ui.h4("Wyniku algorytmu XGBoost na cechach z Boruty:"),
                        ui.output_table("xgb_metrics_with_boruta")
                    )
                ),
                ui.card(
                    ui.card_header("Wykresy cech istotnych"),
                    ui.output_image("boruta_plot")
                ),
            ),
            ui.nav_panel(
                "Porównanie modelów",
                ui.tags.div(
                    ui.tags.div(
                        ui.h4("Wyniku obydwu algorytmów:"),
                        ui.output_table("xgb_boruta_metrics_compared")
                    )
                ),
                ui.card(
                    ui.card_header("Wykresy parametrów"),
                    ui.output_image("comparison_plot")
                ),
            ),
           title=ui.tags.span(
            "Metryki Modelu",
            ui.output_ui("filename")
        ),
        ),
        {"class": "bslib-page-dashboard"},
    ),

      ui.nav_panel(
        "Wyświetl Dane",

        ui.card(ui.output_data_frame("data")),
        {"class": "bslib-page-dashboard"},
    ),

    sidebar=ui.sidebar(
        ui.input_file("file", "Wybierz plik CSV"),
        ui.h5("Parametry uczenia"),
        ui.input_numeric("n_estimators", "Liczba iteracji", value=100, min=10, max=1000),
        ui.input_numeric("learning_rate", "Learning rate", value=0.01, min=0.001, max=1, step=0.01),
        ui.input_numeric("col_sample", "Próbkowanie kolumn", value=0.65, min=0.1, max=1, step=0.05),
        ui.h5("Metryki selekcjonowane"),
        ui.input_numeric("treshold", "Treshold", value=0.5, min=0.1, max=0.95, step=0.05),
        ui.input_numeric("max_boruta", "Ranga od (Boruta)", value=1, min=1, max=100),
        ui.input_numeric("min_boruta", "Ranga do (Boruta)", value=5, min=1, max=100),
    ),
    title="Dashboard wyników",
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc()
    def result():
        if not input.file():
            return None
        path = input.file()[0]["datapath"]
        n_estimators = input.n_estimators()
        learning_rate = input.learning_rate()
        importance_type = input.importance_type()
        col_sample = input.col_sample()
        treshold = input.treshold()

        max_boruta = input.max_boruta()
        min_boruta = input.min_boruta()
        
        return results_data(path, n_estimators=n_estimators, learning_rate=learning_rate, col_sample= col_sample, importance_type = importance_type, treshold = treshold, max_boruta = max_boruta , min_boruta = min_boruta)

    @render.text
    def filename():
        if not input.file():
            return ""
        return  f'Zbiór danych: {input.file()[0]["name"]}'

    @render.table
    def xgb_all_metrics_table():
        r = result()
        if not r:
            return None
        
        return pd.DataFrame([{
            "Accuracy": round(r["data_xgb_metrics"]["accuracy"], 5),
            "Precision": round(r["data_xgb_metrics"]["precision"], 5),
            "Recall": round(r["data_xgb_metrics"]["recall"], 5),
            "F1": round(r["data_xgb_metrics"]["f1"], 5),
            "AUC": round(r["data_xgb_metrics"]["roc_auc_score"], 5),
            "Log Loss": round(r["data_xgb_metrics"]["logloss"], 5),
        }])
    
    
    # @render.table
    # def xgb_auc_selection_table():
    #     r = result()
    #     if not r:
    #         return None
        
    #     return pd.DataFrame({
    #         "Typ selekcji": ["Cały zbiór", "Średnia", "Threshold", "Szum", "Mediana"],
    #         "AUC": [
    #             round(r['data_xgb_selection']["auc_all"], 5),
    #             round(r['data_xgb_selection']["auc_mean"], 5),
    #             round(r['data_xgb_selection']["auc_threshold"], 5),
    #             round(r['data_xgb_selection']["auc_noise"], 5),
    #             round(r['data_xgb_selection']["auc_median"], 5)
    #         ]
    #     })
    
    @render.table
    def xgb_all_metrics_selection_table():
        r = result()
        if not r or "data_xgb_selection_all" not in r:
            return pd.DataFrame(columns=["Typ selekcji","Accuracy", "F1", "AUC", "Precision", "Recall", "Logloss"])

        selection_labels = {
            "mean": "Średnia",
            "threshold": "Threshold",
            "noise": "Szum",
            "median": "Mediana"
        }
        
        data = []
        for key, label in selection_labels.items():
            metrics = r["data_xgb_selection_all"].get(key)
            if metrics:
                data.append({
                    "Typ selekcji": label,
                    "Accuracy": round(metrics.get("accuracy", 0), 5),
                    "F1": round(metrics.get("f1", 0), 5),
                    "Precision": round(metrics.get("precision", 0), 5),
                    "Recall": round(metrics.get("recall", 0), 5),
                    "AUC": round(metrics.get("roc_auc_score", 0), 5),
                    "Log Loss": round(metrics.get("logloss", 0), 5),
                })

        df = pd.DataFrame(data)
        return df
    

    @render.table
    def xgb_metrics_with_boruta():
        r = result()
        if not r:
            return None
        
        return pd.DataFrame([{
            "Accuracy": round(r["xgb_data_with_boruta"]["accuracy"], 5),
            "Precision": round(r["xgb_data_with_boruta"]["precision"], 5),
            "Recall": round(r["xgb_data_with_boruta"]["recall"], 5),
            "F1": round(r["xgb_data_with_boruta"]["f1"], 5),
            "Log Loss": round(r["xgb_data_with_boruta"]["logloss"], 5),
            "AUC": round(r["xgb_data_with_boruta"]['roc_auc_score'], 5)
        }])
    
    @render.table
    def xgb_boruta_metrics_compared():
        r = result()
        if not r or "data_xgb_selection_all" not in r or "xgb_data_with_boruta" not in r:
            return pd.DataFrame(columns=["Typ selekcji", "Accuracy", "F1", "AUC", "Precision", "Recall", "Logloss"])

        selection_labels = {
            "mean": "Średnia",
            "threshold": "Threshold",
            "noise": "Szum",
            "median": "Mediana"
        }


        best_key = None
        best_acc = -1
        best_metrics = None

        for key in selection_labels:
            metrics = r["data_xgb_selection_all"].get(key)
            if metrics and metrics.get("accuracy", 0) > best_acc:
                best_acc = metrics["accuracy"]
                best_key = key
                best_metrics = metrics

        rows = []


        rows.append({
            "Typ selekcji": "XGBoost – wszystkie cechy",
            "Accuracy": round(r["data_xgb_metrics"]["accuracy"], 5),
            "Precision": round(r["data_xgb_metrics"]["precision"], 5),
            "Recall": round(r["data_xgb_metrics"]["recall"], 5),
            "F1": round(r["data_xgb_metrics"]["f1"], 5),
            "AUC": round(r["data_xgb_metrics"]["roc_auc_score"], 5),
            "Logloss": round(r["data_xgb_metrics"]["logloss"], 5),
       })
        

        if best_metrics:
            rows.append({
                "Typ selekcji": f"XGBoost – najlepsze cechy ({selection_labels[best_key]})",
                "Accuracy": round(best_metrics.get("accuracy", 0), 5),
                "F1": round(best_metrics.get("f1", 0), 5),
                "Precision": round(best_metrics.get("precision", 0), 5),
                "Recall": round(best_metrics.get("recall", 0), 5),
                "AUC": round(best_metrics.get("roc_auc_score", 0), 5),
                "Logloss": round(best_metrics.get("logloss", 0), 5),
            })

      
        boruta = r["xgb_data_with_boruta"]
        rows.append({
            "Typ selekcji": "XGBoost – Boruta (top cechy)",
            "Accuracy": round(boruta.get("accuracy", 0), 5),
            "F1": round(boruta.get("f1", 0), 5),
            "Precision": round(boruta.get("precision", 0), 5),
            "Recall": round(boruta.get("recall", 0), 5),
            "AUC": round(boruta.get("roc_auc_score", 0), 5),
            "Logloss": round(boruta.get("logloss", 0), 5),
        })

        return pd.DataFrame(rows)
        
        
    @render.plot
    def gain_plot():
        r = result()
        if not r or "gain_plot_xgb" not in r:
            return None
        return r["gain_plot_xgb"]
    
    @render.plot
    def boruta_plot():
        r = result()
        if not r or "plot_boruta" not in r:
            return None
        return r["plot_boruta"]
    
    @render.plot
    def comparison_plot():
        r = result()
        if not r or "comparison_plot" not in r:
            return None
        return r["comparison_plot"]
    
    @render.ui
    def importance_summary():
        r = result()
        if not r:
            return ""
        return ui.tags.div(
        f"Znaleziono: {r['finded_importance_xgb']} z {r['number_of_importance']}"
    )

    @render.data_frame
    def data():
        r = result()
        return r["df"] if r else None


app = App(app_ui, server)
