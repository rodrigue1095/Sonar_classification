
# ============================================================
# APPLICATION STREAMLIT : DATA SCIENCE ET MACHINE LEARNING
# Jeu de données : SONAR (classification M / R)
# Fichier : app.py
# ============================================================

from pathlib import Path
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. CONFIGURATION DE L'APPLICATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sonar - Data Science et Machine Learning",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

SEED_DEFAULT = 7


# ------------------------------------------------------------
# 2. FONCTIONS DE CHARGEMENT ET DE PREPARATION DES DONNEES
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_dataset(file_bytes, has_header):
    """Lit un CSV avec ou sans ligne d'en-tête."""
    if has_header:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=None,
            engine="python"
        )
    else:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=None,
            engine="python",
            header=None
        )

        # Pour Sonar : 60 variables numériques + une cible
        if df.shape[1] == 61:
            df.columns = [
                *[f"V{i}" for i in range(1, 61)],
                "Classe"
            ]
        else:
            df.columns = [
                *[f"Variable_{i}" for i in range(1, df.shape[1])],
                "Classe"
            ]

    return df


def prepare_data(df, target_col):
    """Prépare les variables explicatives et la cible."""
    data = df.copy()

    # Retirer les lignes sans classe connue
    data = data.dropna(subset=[target_col])

    y = data[target_col].astype(str).str.strip()
    X = data.drop(columns=[target_col]).copy()

    # Convertir les variables en nombres
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Écarter les variables totalement vides
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError(
            "Aucune variable numérique exploitable n'a été trouvée."
        )

    if y.nunique() < 2:
        raise ValueError(
            "La variable cible doit contenir au moins deux classes."
        )

    return X, y


# ------------------------------------------------------------
# 3. DEFINITION DES ALGORITHMES
# ------------------------------------------------------------

def build_models(seed=7, standardize=True):

    algorithms = {
        "Régression logistique": LogisticRegression(
            max_iter=2000,
            random_state=seed
        ),

        "LDA": LinearDiscriminantAnalysis(),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Arbre de décision": DecisionTreeClassifier(
            random_state=seed
        ),

        "Naive Bayes": GaussianNB(),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=seed
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=seed
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=seed
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=seed
        )
    }

    tree_models = {
        "Arbre de décision",
        "AdaBoost",
        "Gradient Boosting",
        "Random Forest",
        "Extra Trees"
    }

    pipelines = {}

    for name, estimator in algorithms.items():

        steps = [
            ("imputation", SimpleImputer(strategy="median"))
        ]

        # Standardisation recommandée pour les modèles sensibles
        # aux échelles des variables
        if standardize and name not in tree_models:
            steps.append(
                ("standardisation", StandardScaler())
            )

        steps.append(("modele", estimator))

        pipelines[name] = Pipeline(steps)

    return pipelines


# ------------------------------------------------------------
# 4. FONCTIONS GRAPHIQUES
# ------------------------------------------------------------

def display_figure(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred):
    labels = sorted(pd.Series(y_true).astype(str).unique())

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")

    display_figure(fig)


# ------------------------------------------------------------
# 5. INTERFACE PRINCIPALE
# ------------------------------------------------------------

def main():

    st.title(
        "📡 Application de Data Science et de Machine Learning"
    )

    st.markdown(
        """
        Cette application permet de réaliser une analyse exploratoire,
        de comparer plusieurs algorithmes de classification, d'évaluer
        leurs performances et de prédire la classe de nouvelles données.
        """
    )

    # --------------------------------------------------------
    # BARRE LATERALE
    # --------------------------------------------------------

    st.sidebar.header("⚙️ Configuration")

    uploaded_file = st.sidebar.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    has_header = st.sidebar.checkbox(
        "La première ligne contient les noms des colonnes",
        value=False,
        help=(
            "Pour le fichier Sonar classique sans en-tête, "
            "laissez cette option décochée."
        )
    )

    test_size = st.sidebar.slider(
        "Proportion du jeu de test (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    ) / 100

    cv_folds = st.sidebar.slider(
        "Nombre de plis de validation croisée",
        min_value=3,
        max_value=10,
        value=5
    )

    seed = st.sidebar.number_input(
        "Graine aléatoire",
        min_value=0,
        max_value=99999,
        value=SEED_DEFAULT
    )

    standardize = st.sidebar.checkbox(
        "Standardiser les variables numériques",
        value=True
    )

    # --------------------------------------------------------
    # CHARGEMENT DU FICHIER
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:
            dataset = read_dataset(
                uploaded_file.getvalue(),
                has_header
            )
            source_name = uploaded_file.name

        except Exception as error:
            st.error(
                f"Erreur lors de la lecture du CSV : {error}"
            )
            st.stop()

    else:

        local_file = Path(__file__).parent / "sonar_data.csv"

        if not local_file.exists():

            st.info(
                "Importez `sonar_data.csv` dans la barre latérale "
                "ou placez ce fichier à côté de `app.py`."
            )

            st.markdown(
                """
                **Format du fichier Sonar classique :**
                - 60 colonnes numériques ;
                - une dernière colonne contenant `M` ou `R` ;
                - 208 observations dans le jeu original.
                """
            )

            st.stop()

        try:
            dataset = read_dataset(
                local_file.read_bytes(),
                has_header
            )
            source_name = local_file.name

        except Exception as error:
            st.error(
                f"Erreur de lecture du fichier local : {error}"
            )
            st.stop()

    if dataset.empty or dataset.shape[1] < 2:
        st.error(
            "Le fichier doit contenir au moins deux colonnes."
        )
        st.stop()

    st.success(
        f"Fichier chargé : {source_name} — "
        f"{dataset.shape[0]} lignes et {dataset.shape[1]} colonnes."
    )

    # --------------------------------------------------------
    # CHOIX DE LA CIBLE
    # --------------------------------------------------------

    target_col = st.sidebar.selectbox(
        "Variable cible à prédire",
        options=list(dataset.columns),
        index=len(dataset.columns) - 1
    )

    try:
        X, y = prepare_data(dataset, target_col)

    except ValueError as error:
        st.error(str(error))
        st.stop()

    # --------------------------------------------------------
    # ONGLETS
    # --------------------------------------------------------

    tab_data, tab_eda, tab_models, tab_prediction = st.tabs(
        [
            "📊 Exploration des données",
            "🔎 Analyse exploratoire",
            "🤖 Machine Learning",
            "🎯 Prédictions"
        ]
    )

    # ========================================================
    # ONGLET 1 : EXPLORATION DES DONNEES
    # ========================================================

    with tab_data:

        st.header("Exploration du jeu de données")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Observations",
            f"{len(dataset)}"
        )

        col2.metric(
            "Variables explicatives",
            f"{X.shape[1]}"
        )

        col3.metric(
            "Classes",
            f"{y.nunique()}"
        )

        col4.metric(
            "Valeurs manquantes",
            f"{dataset.isna().sum().sum()}"
        )

        st.subheader("Aperçu des données")

        st.dataframe(
            dataset.head(20),
            use_container_width=True
        )

        st.subheader("Types de variables")

        types_df = pd.DataFrame({
            "Variable": dataset.columns.astype(str),
            "Type": dataset.dtypes.astype(str).values,
            "Valeurs manquantes": dataset.isna().sum().values,
            "Valeurs distinctes": dataset.nunique().values
        })

        st.dataframe(
            types_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Statistiques descriptives")

        st.dataframe(
            X.describe().T.round(4),
            use_container_width=True
        )

        st.subheader("Répartition de la variable cible")

        distribution = (
            y.value_counts()
            .rename_axis("Classe")
            .reset_index(name="Effectif")
        )

        distribution["Pourcentage"] = (
            100 * distribution["Effectif"]
            / distribution["Effectif"].sum()
        ).round(2)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True
            )

        with col2:
            fig, ax = plt.subplots()

            y.value_counts().plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Distribution des classes")
            ax.set_xlabel("Classe")
            ax.set_ylabel("Effectif")
            ax.tick_params(axis="x", rotation=0)

            display_figure(fig)

        st.download_button(
            "Télécharger les données nettoyées en CSV",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name="donnees_sonar.csv",
            mime="text/csv"
        )

    # ========================================================
    # ONGLET 2 : ANALYSE EXPLORATOIRE
    # ========================================================

    with tab_eda:

        st.header("Analyse exploratoire des données")

        numeric_features = X.columns.tolist()

        selected_features = st.multiselect(
            "Choisir les variables à visualiser",
            options=numeric_features,
            default=numeric_features[
                :min(6, len(numeric_features))
            ]
        )

        if selected_features:

            st.subheader("Histogrammes et distributions")

            for start in range(0, len(selected_features), 2):

                cols = st.columns(2)

                for offset in range(2):

                    idx = start + offset

                    if idx >= len(selected_features):
                        break

                    feature = selected_features[idx]

                    with cols[offset]:

                        fig, ax = plt.subplots()

                        sns.histplot(
                            X[feature].dropna(),
                            kde=True,
                            ax=ax
                        )

                        ax.set_title(
                            f"Distribution de {feature}"
                        )

                        display_figure(fig)

            st.subheader("Diagrammes en boîtes")

            fig, ax = plt.subplots(
                figsize=(
                    max(8, len(selected_features)),
                    5
                )
            )

            sns.boxplot(
                data=X[selected_features],
                ax=ax
            )

            ax.set_title("Détection visuelle des valeurs extrêmes")
            ax.tick_params(axis="x", rotation=45)

            display_figure(fig)

        st.subheader("Matrice de corrélation")

        max_features = min(60, len(numeric_features))

        n_corr = st.slider(
            "Nombre de variables à afficher",
            min_value=min(5, max_features),
            max_value=max_features,
            value=min(20, max_features)
        )

        corr = X[numeric_features[:n_corr]].corr()

        fig, ax = plt.subplots(figsize=(12, 9))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            ax=ax
        )

        ax.set_title("Matrice de corrélation")
        display_figure(fig)

        st.caption(
            "La corrélation mesure une association statistique ; "
            "elle ne prouve pas une relation de causalité."
        )

    # ========================================================
    # ONGLET 3 : MACHINE LEARNING
    # ========================================================

    with tab_models:

        st.header("Entraînement et comparaison des modèles")

        st.write(
            "Les modèles sont comparés par validation croisée "
            "stratifiée sur le jeu d'entraînement. Le jeu de test "
            "est conservé pour une évaluation finale."
        )

        class_counts = y.value_counts()

        stratify_y = (
            y if class_counts.min() >= 2
            else None
        )

        if stratify_y is None:
            st.warning(
                "Certaines classes contiennent trop peu "
                "d'observations pour une séparation stratifiée."
            )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=int(seed),
                stratify=stratify_y
            )

        except ValueError as error:
            st.error(
                f"Impossible de séparer les données : {error}"
            )
            st.stop()

        effective_folds = min(
            int(cv_folds),
            int(y_train.value_counts().min())
        )

        if effective_folds < 2:
            st.error(
                "Il faut au moins deux observations dans chaque "
                "classe du jeu d'entraînement."
            )
            st.stop()

        if effective_folds != cv_folds:
            st.warning(
                f"Le nombre de plis a été réduit à {effective_folds} "
                "pour tenir compte de la taille des classes."
            )

        models = build_models(
            seed=int(seed),
            standardize=standardize
        )

        if st.button(
            "🚀 Lancer la comparaison des algorithmes",
            type="primary"
        ):

            cv = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=int(seed)
            )

            results = []
            fitted_models = {}

            progress = st.progress(0)
            status = st.empty()

            for i, (name, model) in enumerate(models.items()):

                status.write(
                    f"Entraînement et évaluation : {name}"
                )

                try:

                    cv_scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring="accuracy",
                        n_jobs=-1
                    )

                    # Entraîner le modèle sur toutes les données
                    # du jeu d'entraînement
                    model.fit(
                        X_train,
                        y_train
                    )

                    test_predictions = model.predict(X_test)

                    results.append({
                        "Modèle": name,
                        "Exactitude CV": cv_scores.mean(),
                        "Écart-type CV": cv_scores.std(),
                        "Exactitude test": accuracy_score(
                            y_test,
                            test_predictions
                        ),
                        "Balanced accuracy": balanced_accuracy_score(
                            y_test,
                            test_predictions
                        ),
                        "F1 macro": f1_score(
                            y_test,
                            test_predictions,
                            average="macro",
                            zero_division=0
                        )
                    })

                    fitted_models[name] = model

                except Exception as error:

                    st.warning(
                        f"Le modèle {name} n'a pas pu être évalué : "
                        f"{error}"
                    )

                progress.progress(
                    (i + 1) / len(models)
                )

            progress.empty()
            status.empty()

            if not results:

                st.error(
                    "Aucun modèle n'a pu être entraîné."
                )

            else:

                results_df = (
                    pd.DataFrame(results)
                    .sort_values(
                        "Exactitude CV",
                        ascending=False
                    )
                    .reset_index(drop=True)
                )

                st.session_state["results"] = results_df
                st.session_state["fitted_models"] = fitted_models

                st.session_state["X_test"] = X_test.copy()
                st.session_state["y_test"] = y_test.copy()
                st.session_state["feature_names"] = X.columns.tolist()

                st.success(
                    "La comparaison des modèles est terminée."
                )

                st.subheader("Tableau comparatif")

                st.dataframe(
                    results_df.style.format({
                        "Exactitude CV": "{:.3f}",
                        "Écart-type CV": "{:.3f}",
                        "Exactitude test": "{:.3f}",
                        "Balanced accuracy": "{:.3f}",
                        "F1 macro": "{:.3f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Graphique comparatif")

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.barplot(
                    data=results_df,
                    x="Exactitude CV",
                    y="Modèle",
                    ax=ax
                )

                ax.set_xlim(0, 1)
                ax.set_xlabel("Exactitude moyenne en validation croisée")
                ax.set_ylabel("Algorithme")
                ax.set_title("Comparaison des algorithmes")

                display_figure(fig)

                st.download_button(
                    "Télécharger les résultats de comparaison",
                    data=results_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="resultats_modeles.csv",
                    mime="text/csv"
                )

        # ----------------------------------------------------
        # EVALUATION DETAILLEE DU MODELE
        # ----------------------------------------------------

        if (
            "results" in st.session_state
            and "fitted_models" in st.session_state
        ):

            st.markdown("---")
            st.subheader("Évaluation détaillée")

            available_models = list(
                st.session_state["fitted_models"].keys()
            )

            chosen_model = st.selectbox(
                "Choisir un modèle entraîné",
                available_models
            )

            model = st.session_state[
                "fitted_models"
            ][chosen_model]

            X_test_saved = st.session_state["X_test"]
            y_test_saved = st.session_state["y_test"]

            predictions = model.predict(X_test_saved)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Exactitude",
                f"{accuracy_score(y_test_saved, predictions):.1%}"
            )

            col2.metric(
                "Balanced accuracy",
                f"{balanced_accuracy_score(y_test_saved, predictions):.1%}"
            )

            col3.metric(
                "Précision macro",
                f"{precision_score(y_test_saved, predictions, average='macro', zero_division=0):.1%}"
            )

            col4.metric(
                "F1 macro",
                f"{f1_score(y_test_saved, predictions, average='macro', zero_division=0):.1%}"
            )

            st.subheader("Matrice de confusion")

            plot_confusion_matrix(
                y_test_saved,
                predictions
            )

            st.subheader("Rapport de classification")

            report = classification_report(
                y_test_saved,
                predictions,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).T

            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )

            st.download_button(
                "Télécharger le rapport de classification",
                data=report_df.to_csv().encode("utf-8"),
                file_name="rapport_classification.csv",
                mime="text/csv"
            )

        # ----------------------------------------------------
        # OPTIMISATION DES HYPERPARAMETRES
        # ----------------------------------------------------

        st.markdown("---")
        st.subheader("Optimisation des hyperparamètres avec GridSearchCV")

        st.write(
            "Cette procédure recherche les paramètres les plus "
            "performants pour un SVM. La standardisation et "
            "l'imputation sont intégrées au pipeline."
        )

        if st.button("🔧 Optimiser le modèle SVM"):

            X_train_opt, X_test_opt, y_train_opt, y_test_opt = (
                train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=int(seed),
                    stratify=y if y.value_counts().min() >= 2 else None
                )
            )

            folds_opt = min(
                int(cv_folds),
                int(y_train_opt.value_counts().min())
            )

            if folds_opt < 2:

                st.error(
                    "Pas assez d'observations pour la validation croisée."
                )

            else:

                svm_pipeline = Pipeline([
                    (
                        "imputation",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "standardisation",
                        StandardScaler()
                    ),
                    (
                        "modele",
                        SVC(probability=True, random_state=int(seed))
                    )
                ])

                param_grid = {
                    "modele__C": [
                        0.1, 0.5, 1.0, 1.5, 2.0, 3.0
                    ],
                    "modele__kernel": [
                        "linear", "rbf", "poly"
                    ],
                    "modele__gamma": [
                        "scale", "auto"
                    ]
                }

                cv_opt = StratifiedKFold(
                    n_splits=folds_opt,
                    shuffle=True,
                    random_state=int(seed)
                )

                with st.spinner(
                    "Recherche des meilleurs paramètres en cours..."
                ):

                    search = GridSearchCV(
                        estimator=svm_pipeline,
                        param_grid=param_grid,
                        scoring="accuracy",
                        cv=cv_opt,
                        n_jobs=-1,
                        refit=True
                    )

                    search.fit(
                        X_train_opt,
                        y_train_opt
                    )

                best_model = search.best_estimator_

                best_predictions = best_model.predict(
                    X_test_opt
                )

                st.success(
                    f"Meilleure exactitude moyenne en validation croisée : "
                    f"{search.best_score_:.3f}"
                )

                st.write(
                    "Meilleurs paramètres :",
                    search.best_params_
                )

                st.metric(
                    "Exactitude sur le jeu de test",
                    f"{accuracy_score(y_test_opt, best_predictions):.1%}"
                )

                st.subheader("Résultats de la recherche")

                cv_results = pd.DataFrame(
                    search.cv_results_
                ).sort_values("rank_test_score")

                st.dataframe(
                    cv_results[
                        [
                            "rank_test_score",
                            "mean_test_score",
                            "std_test_score",
                            "params"
                        ]
                    ].head(20),
                    use_container_width=True,
                    hide_index=True
                )

                st.session_state["best_model"] = best_model
                st.session_state["best_features"] = X.columns.tolist()

                st.success(
                    "Le modèle SVM optimisé est maintenant disponible "
                    "dans l'onglet Prédictions."
                )

    # ========================================================
    # ONGLET 4 : PREDICTIONS
    # ========================================================

    with tab_prediction:

        st.header("Prédire la classe d'une nouvelle observation")

        st.write(
            "Entraînez d'abord un modèle dans l'onglet "
            "« Machine Learning ». Vous pourrez ensuite importer "
            "un fichier CSV ou saisir manuellement les valeurs."
        )

        trained_models = st.session_state.get(
            "fitted_models",
            {}
        )

        if trained_models:

            model_options = list(trained_models.keys())

            if "best_model" in st.session_state:
                model_options.append("SVM optimisé")

            chosen_prediction_model = st.selectbox(
                "Modèle utilisé pour la prédiction",
                model_options
            )

            if chosen_prediction_model == "SVM optimisé":

                prediction_model = st.session_state["best_model"]

            else:

                prediction_model = trained_models[
                    chosen_prediction_model
                ]

            # ------------------------------------------------
            # PREDICTIONS DEPUIS UN FICHIER CSV
            # ------------------------------------------------

            st.subheader("Méthode A : importer plusieurs observations")

            prediction_file = st.file_uploader(
                "CSV des nouvelles observations",
                type=["csv"],
                key="prediction_upload"
            )

            if prediction_file is not None:

                try:

                    new_data = pd.read_csv(
                        io.BytesIO(prediction_file.getvalue()),
                        sep=None,
                        engine="python"
                    )

                    expected_features = (
                        st.session_state["feature_names"]
                    )

                    # Si le nombre de colonnes correspond, mais que
                    # les noms diffèrent, reprendre l'ordre attendu.
                    if set(new_data.columns) == set(expected_features):

                        new_data = new_data[expected_features]

                    elif new_data.shape[1] == len(expected_features):

                        new_data.columns = expected_features

                    else:

                        st.error(
                            f"Le fichier doit contenir exactement "
                            f"{len(expected_features)} variables prédictives "
                            "et ne doit pas contenir la colonne cible."
                        )

                        new_data = None

                    if new_data is not None:

                        for col in new_data.columns:
                            new_data[col] = pd.to_numeric(
                                new_data[col],
                                errors="coerce"
                            )

                        new_predictions = prediction_model.predict(
                            new_data
                        )

                        output = new_data.copy()

                        output["Classe prédite"] = new_predictions

                        if hasattr(
                            prediction_model,
                            "predict_proba"
                        ):

                            probabilities = (
                                prediction_model.predict_proba(
                                    new_data
                                )
                            )

                            for i, class_name in enumerate(
                                prediction_model.classes_
                            ):

                                output[
                                    f"Probabilité {class_name}"
                                ] = probabilities[:, i]

                        st.dataframe(
                            output,
                            use_container_width=True
                        )

                        st.download_button(
                            "Télécharger les prédictions",
                            data=output.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="predictions_sonar.csv",
                            mime="text/csv"
                        )

                except Exception as error:

                    st.error(
                        f"Erreur pendant la prédiction : {error}"
                    )

            # ------------------------------------------------
            # PREDICTION MANUELLE
            # ------------------------------------------------

            st.markdown("---")
            st.subheader("Méthode B : saisir une observation")

            feature_names = st.session_state["feature_names"]

            default_values = X.median(numeric_only=True)

            with st.form("manual_prediction_form"):

                st.caption(
                    "Les valeurs initiales correspondent aux médianes "
                    "du jeu de données. Remplacez-les par les mesures "
                    "de votre nouvel échantillon."
                )

                values = {}

                for start in range(0, len(feature_names), 3):

                    cols = st.columns(3)

                    for j, feature in enumerate(
                        feature_names[start:start + 3]
                    ):

                        with cols[j]:

                            values[feature] = st.number_input(
                                feature,
                                value=float(default_values[feature]),
                                format="%.6f",
                                key=f"manual_{feature}"
                            )

                submit_prediction = st.form_submit_button(
                    "🎯 Prédire la classe"
                )

            if submit_prediction:

                new_observation = pd.DataFrame(
                    [values],
                    columns=feature_names
                )

                predicted_class = prediction_model.predict(
                    new_observation
                )[0]

                st.success(
                    f"Classe prédite : **{predicted_class}**"
                )

                if hasattr(
                    prediction_model,
                    "predict_proba"
                ):

                    probabilities = (
                        prediction_model.predict_proba(
                            new_observation
                        )[0]
                    )

                    probability_df = pd.DataFrame({
                        "Classe": prediction_model.classes_,
                        "Probabilité": probabilities
                    }).sort_values(
                        "Probabilité",
                        ascending=False
                    )

                    st.subheader(
                        "Probabilités estimées par classe"
                    )

                    st.dataframe(
                        probability_df.style.format({
                            "Probabilité": "{:.2%}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

        else:

            st.warning(
                "Aucun modèle n'est encore entraîné. "
                "Ouvrez l'onglet Machine Learning et lancez "
                "la comparaison des algorithmes."
            )

    # --------------------------------------------------------
    # PIED DE PAGE
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        "Application pédagogique de Data Science et Machine Learning. "
        "Les performances observées ne garantissent pas les résultats "
        "sur de nouvelles données. Validez toujours le modèle selon "
        "le contexte d'utilisation."
    )


# ------------------------------------------------------------
# 6. POINT D'ENTREE
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
