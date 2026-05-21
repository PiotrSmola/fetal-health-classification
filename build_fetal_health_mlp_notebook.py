from pathlib import Path
from textwrap import dedent

import nbformat as nbf


OUTPUT_PATH = Path("fetal_health_mlp_project2.ipynb")


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.14.3",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    },
}

nb["cells"] = [
    markdown(
        """
        # Projekt 2: siec neuronowa MLP dla klasyfikacji fetal health

        **Cel:** sprawdzic, czy mala siec neuronowa typu **MLP** potrafi klasyfikowac
        stan zdrowia plodu na podstawie cech tabelarycznych z pomiarow CTG.

        Notebook jest pierwsza wersja projektu 2 na tym samym zbiorze co projekt 1.
        Obecny CSV zawiera gotowe cechy liczbowe, a nie surowy sygnal czasowy CTG,
        dlatego naturalna architektura startowa to MLP. Modele CNN, RNN i LSTM
        mialyby wiecej sensu dopiero dla surowych sekwencji pomiarowych.
        """
    ),
    markdown(
        """
        ## Plan eksperymentu

        1. Wczytanie i szybka kontrola zbioru.
        2. Stratyfikowany podzial train/test.
        3. Selekcja silnie skorelowanych cech liczona tylko na train.
        4. Skalowanie cech i trening malej sieci MLP.
        5. Porownanie baseline oraz wariantu z `SMOTE`.
        6. Tuning malej architektury MLP przez cross-validation.
        7. Ewaluacja na odlozonym zbiorze testowym i porownanie z projektem 1.

        Glowna metryka to **macro F1**, bo klasy sa wyraznie niezbalansowane.
        """
    ),
    code(
        """
        import warnings

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            ConfusionMatrixDisplay,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        warnings.filterwarnings("ignore")

        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 110

        CLASS_LABELS = {
            1: "Normal",
            2: "Suspect",
            3: "Pathological",
        }

        print("Biblioteki zaladowane.")
        """
    ),
    markdown("## 1. Wczytanie i kontrola danych"),
    code(
        """
        df = pd.read_csv("fetal_health.csv")
        df.columns = df.columns.str.replace(" ", "_", regex=False)
        df["fetal_health"] = df["fetal_health"].astype(int)

        print(f"Wymiary zbioru: {df.shape[0]} rekordow, {df.shape[1]} kolumn")
        print(f"Brakujace wartosci: {df.isna().sum().sum()}")
        print(f"Duplikaty: {df.duplicated().sum()}")
        df.head()
        """
    ),
    code(
        """
        feature_columns = df.columns.drop("fetal_health")
        feature_summary = pd.DataFrame(
            {
                "dtype": df[feature_columns].dtypes.astype(str),
                "min": df[feature_columns].min(),
                "median": df[feature_columns].median(),
                "max": df[feature_columns].max(),
                "std": df[feature_columns].std(),
            }
        ).round(4)

        feature_summary
        """
    ),
    code(
        """
        class_counts = df["fetal_health"].value_counts().sort_index()
        class_percent = (class_counts / len(df) * 100).round(1)
        class_table = pd.DataFrame(
            {
                "label": [CLASS_LABELS[i] for i in class_counts.index],
                "records": class_counts,
                "share_percent": class_percent,
            }
        )

        display(class_table)

        ax = sns.barplot(
            data=class_table,
            x="label",
            y="records",
            hue="label",
            palette=["#2a9d8f", "#f4a261", "#e76f51"],
            legend=False,
        )
        for row_index, row in class_table.reset_index(drop=True).iterrows():
            ax.text(
                row_index,
                row["records"] + 20,
                f'{row["share_percent"]:.1f}%',
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_title("Rozklad klas w zbiorze fetal health")
        ax.set_xlabel("Klasa")
        ax.set_ylabel("Liczba rekordow")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 2. Podzial danych i selekcja cech

        Siec MLP jest wrazliwa na skale cech, wiec skalowanie trafi pozniej do
        pipeline. Najpierw odkladamy test i usuwamy wybrane silnie skorelowane
        cechy na podstawie macierzy korelacji liczonej **wylacznie na train**.
        """
    ),
    code(
        """
        X = df.drop(columns="fetal_health")
        y = df["fetal_health"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        corr_train = X_train.corr().abs()
        upper_triangle = corr_train.where(
            np.triu(np.ones(corr_train.shape), k=1).astype(bool)
        )
        correlated_to_drop = [
            column for column in upper_triangle.columns
            if (upper_triangle[column] > 0.85).any()
        ]

        X_train_selected = X_train.drop(columns=correlated_to_drop)
        X_test_selected = X_test.drop(columns=correlated_to_drop)

        print(f"Train: {X_train_selected.shape}")
        print(f"Test:  {X_test_selected.shape}")
        print(f"Usuniete cechy: {correlated_to_drop}")
        print(f"Pozostale cechy ({X_train_selected.shape[1]}):")
        print(X_train_selected.columns.tolist())
        print("\\nRozklad klas w train:")
        print(y_train.value_counts().sort_index())
        print("\\nRozklad klas w test:")
        print(y_test.value_counts().sort_index())
        """
    ),
    markdown(
        """
        ## 3. Baseline: mala siec MLP

        `MLPClassifier` implementuje wielowarstwowy perceptron z warstwami gestymi.
        Baseline ma dwie male warstwy ukryte `(64, 32)`, aktywacje ReLU,
        regularyzacje `alpha` oraz early stopping.
        """
    ),
    code(
        """
        def make_mlp(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3):
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=alpha,
                batch_size=32,
                learning_rate_init=learning_rate_init,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=RANDOM_STATE,
            )


        mlp_baseline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", make_mlp()),
            ]
        )
        mlp_smote = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", make_mlp()),
            ]
        )

        mlp_baseline.fit(X_train_selected, y_train)
        mlp_smote.fit(X_train_selected, y_train)

        baseline_predictions = mlp_baseline.predict(X_test_selected)
        smote_predictions = mlp_smote.predict(X_test_selected)
        """
    ),
    code(
        """
        def metric_row(model_name, y_true, y_pred):
            return {
                "Model": model_name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision_macro": precision_score(y_true, y_pred, average="macro"),
                "Recall_macro": recall_score(y_true, y_pred, average="macro"),
                "F1_macro": f1_score(y_true, y_pred, average="macro"),
                "F1_Normal": f1_score(y_true, y_pred, average=None, labels=[1, 2, 3])[0],
                "F1_Suspect": f1_score(y_true, y_pred, average=None, labels=[1, 2, 3])[1],
                "F1_Pathological": f1_score(y_true, y_pred, average=None, labels=[1, 2, 3])[2],
            }


        baseline_comparison = pd.DataFrame(
            [
                metric_row("MLP baseline", y_test, baseline_predictions),
                metric_row("MLP + SMOTE", y_test, smote_predictions),
            ]
        ).set_index("Model")

        baseline_comparison.round(4)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, pipeline, title in [
            (axes[0], mlp_baseline, "MLP baseline"),
            (axes[1], mlp_smote, "MLP + SMOTE"),
        ]:
            model = pipeline.named_steps["model"]
            ax.plot(model.loss_curve_, label="train loss", color="#264653")
            if hasattr(model, "validation_scores_") and model.validation_scores_:
                second_axis = ax.twinx()
                second_axis.plot(
                    model.validation_scores_,
                    label="validation score",
                    color="#e76f51",
                    alpha=0.75,
                )
                second_axis.set_ylabel("Validation score")
            ax.set_title(title)
            ax.set_xlabel("Epoka / iteracja")
            ax.set_ylabel("Loss")

        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 4. Tuning architektury i regularyzacji

        Szukamy tylko malych sieci. Przy 2126 rekordach zbyt duza architektura
        latwo nauczy sie szczegolow train zamiast stabilnych zaleznosci.

        `SMOTE` pozostaje w pipeline, wiec nowe probki sa generowane tylko
        wewnatrz foldow treningowych cross-validation.
        """
    ),
    code(
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        tuning_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", make_mlp()),
            ]
        )

        param_grid = {
            "model__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [5e-4, 1e-3],
        }

        mlp_grid = GridSearchCV(
            estimator=tuning_pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=1,
        )
        mlp_grid.fit(X_train_selected, y_train)

        print(f"Najlepsze parametry: {mlp_grid.best_params_}")
        print(f"Najlepszy wynik CV macro F1: {mlp_grid.best_score_:.4f}")
        """
    ),
    code(
        """
        cv_results = (
            pd.DataFrame(mlp_grid.cv_results_)
            .sort_values("rank_test_score")
            [
                [
                    "rank_test_score",
                    "mean_test_score",
                    "std_test_score",
                    "mean_train_score",
                    "param_model__hidden_layer_sizes",
                    "param_model__alpha",
                    "param_model__learning_rate_init",
                ]
            ]
            .head(10)
        )

        cv_results.round(4)
        """
    ),
    markdown("## 5. Ewaluacja najlepszego MLP na tescie"),
    code(
        """
        best_mlp = mlp_grid.best_estimator_
        best_predictions = best_mlp.predict(X_test_selected)

        tuned_metrics = pd.DataFrame(
            [metric_row("MLP tuned + SMOTE", y_test, best_predictions)]
        ).set_index("Model")
        tuned_metrics.round(4)
        """
    ),
    code(
        """
        target_names = [f"{class_id} - {CLASS_LABELS[class_id]}" for class_id in [1, 2, 3]]

        print("Classification report - najlepszy MLP")
        print(classification_report(y_test, best_predictions, target_names=target_names))

        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            best_predictions,
            labels=[1, 2, 3],
            display_labels=target_names,
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title("Confusion matrix - najlepszy MLP")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        best_model = best_mlp.named_steps["model"]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(best_model.loss_curve_, color="#264653", label="train loss")
        ax.set_xlabel("Epoka / iteracja")
        ax.set_ylabel("Loss")
        ax.set_title("Krzywa uczenia najlepszego MLP")

        if hasattr(best_model, "validation_scores_") and best_model.validation_scores_:
            validation_axis = ax.twinx()
            validation_axis.plot(
                best_model.validation_scores_,
                color="#e76f51",
                label="validation score",
                alpha=0.75,
            )
            validation_axis.set_ylabel("Validation score")

        plt.tight_layout()
        plt.show()

        print(f"Liczba iteracji najlepszego MLP: {best_model.n_iter_}")
        print(
            "Najlepszy validation score w early stopping: "
            f"{getattr(best_model, 'best_validation_score_', None)}"
        )
        """
    ),
    markdown(
        """
        ## 6. Porownanie z punktem odniesienia z projektu 1

        Ponizsze metryki RF i XGBoost sa przepisane z pierwszego notebooka,
        gdzie te modele byly ewaluowane na tym samym podziale `random_state=42`
        i na tym samym zbiorze testowym.
        """
    ),
    code(
        """
        project1_reference = pd.DataFrame(
            [
                {
                    "Model": "Random Forest - projekt 1",
                    "Accuracy": 0.9319,
                    "Precision_macro": 0.8683,
                    "Recall_macro": 0.8857,
                    "F1_macro": 0.8768,
                    "F1_Normal": 0.9636,
                    "F1_Suspect": 0.7934,
                    "F1_Pathological": 0.8732,
                },
                {
                    "Model": "XGBoost - projekt 1",
                    "Accuracy": 0.9460,
                    "Precision_macro": 0.8973,
                    "Recall_macro": 0.8863,
                    "F1_macro": 0.8898,
                    "F1_Normal": 0.9746,
                    "F1_Suspect": 0.8182,
                    "F1_Pathological": 0.8767,
                },
            ]
        ).set_index("Model")

        final_comparison = pd.concat(
            [
                project1_reference,
                baseline_comparison,
                tuned_metrics,
            ]
        )

        final_comparison.round(4).sort_values("F1_macro", ascending=False)
        """
    ),
    markdown(
        """
        ## 7. Wnioski robocze

        - Zbior nadaje sie do malego MLP, bo wszystkie cechy wejsciowe sa liczbowe
          i tworza klasyfikacje wieloklasowa.
        - Najwazniejsze ryzyka eksperymentu to mala liczba obserwacji dla klas
          `Suspect` i `Pathological` oraz mozliwe przeuczenie za duzej sieci.
        - Dla danych tabelarycznych modele drzewiaste z projektu 1 sa mocnym
          punktem odniesienia. MLP nie musi ich pobic, aby eksperyment byl
          merytorycznie wartosciowy.
        - Jesli wyniki MLP okaza sie za slabe albo projekt ma wymagac innej
          architektury niz gesty perceptron, kolejny krok to wiekszy zbior lub
          surowy sygnal/sekwencja zamiast gotowych cech tabelarycznych.
        """
    ),
]

nbf.write(nb, OUTPUT_PATH)
print(f"Utworzono {OUTPUT_PATH.resolve()}")
