from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from logic import results_data
import pandas as pd
app_ui = ui.page_navbar(
    ui.nav_spacer(),
     ui.nav_panel(
        "Wyniki modelu",
        ui.navset_card_underline(
            ui.nav_panel(
                "XGB",
               ui.tags.div(
                    ui.h4("Cały Zbiór:"),
                    ui.output_table("xgb_all_metrics_table"),
                    ui.h4("Zbiór selekcjonwany:"),
                    ui.output_table("xgb_all_metrics_selection_table")
                ),
                ui.tags.div(
                    ui.tags.div(
                        ui.h4("Selekcjonowany Zbiór:"),
                        ui.output_table("xgb_auc_selection_table")
                    )
                ),
                ui.card(
                    ui.card_header("Wykresy cech istotnych"),
                    ui.input_select("importance_type", "Rodzaj hyperparametru:", {
                        "gain": "Gain",
                        "combined_score": "Combined Score"
                    }),
                    ui.output_image("gain_plot"),
                    ui.output_ui("importance_summary")
                )
            ),
            ui.nav_panel(
                "Boruta",
               ui.tags.div(
                    ui.h4("Cały Zbiór:"),
                    # ui.output_table("xgb_all_metrics_table")
                ),
                ui.tags.div(
                    ui.tags.div(
                        ui.h4("Selekcjonowany Zbiór:"),
                        # ui.output_table("xgb_auc_selection_table")
                    )
                ),
                ui.card(
                    ui.card_header("Wykresy cech istotnych"),
                    # ui.output_image("gain_plot")
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
        ui.input_numeric("treshold", "Treshold", value=0.5, min=0.1, max=0.95, step=0.05)
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
        return results_data(path, n_estimators=n_estimators, learning_rate=learning_rate, col_sample= col_sample, importance_type = importance_type, treshold = treshold)

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
            "Log Loss": round(r["data_xgb_metrics"]["logloss"], 5),
        }])
    
    @render.text
    def xgb_auc_score():
        r = result()
        return r["best_auc"] if r else ""
    
    @render.table
    def xgb_auc_selection_table():
        r = result()
        if not r:
            return None
        
        return pd.DataFrame({
            "Typ selekcji": ["Cały zbiór", "Średnia", "Threshold", "Szum", "Mediana"],
            "AUC": [
                round(r['data_xgb_selection']["auc_all"], 5),
                round(r['data_xgb_selection']["auc_mean"], 5),
                round(r['data_xgb_selection']["auc_threshold"], 5),
                round(r['data_xgb_selection']["auc_noise"], 5),
                round(r['data_xgb_selection']["auc_median"], 5)
            ]
        })
    
    @render.data_frame
    def xgb_all_metrics_selection_table():
        r = result()
        if not r or "data_xgb_selection_all" not in r:
            return None

        selection_labels = {
            "mean": "Średnia",
            "threshold": "Threshold",
            "noise": "Szum",
            "median": "Mediana"
        }
        print(r["data_xgb_selection_all"])
        data = []
        for key, label in selection_labels.items():
            data.append({
                "Typ selekcji": label,
                "F1": round(r["data_xgb_selection_all"][key]["f1"], 5),
                "Precision": round(r["data_xgb_selection_all"][key]["precision"], 5),
                "Recall": round(r["data_xgb_selection_all"][key]["recall"], 5)
            })

        return pd.DataFrame(data)
        
    @render.plot
    def gain_plot():
        r = result()
        if not r or "gain_plot_xgb" not in r:
            return None
        return r["gain_plot_xgb"]
    
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
