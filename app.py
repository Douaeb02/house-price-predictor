# app.py
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# --- Config ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Chargement des artefacts ---
def load_artifact(path, loader="joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}. Exécute d'abord train_model.py pour générer les artefacts.")
    if loader == "joblib":
        return joblib.load(path)
    elif loader == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("loader doit être 'joblib' ou 'pickle'")

try:
    model = load_artifact(MODEL_PATH, loader="joblib")
    scaler = load_artifact(SCALER_PATH, loader="joblib")
    feature_names = load_artifact(FEATURES_PATH, loader="pickle")
except Exception as e:
    raise

# --- Plots ---
def plot_feature_importance():
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        df_imp = pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
    else:
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x="importance", y="feature", data=df_imp, palette="Blues_r")
    plt.title("Feature Importances")
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "feature_importances.png")
    plt.savefig(fname, dpi=240, bbox_inches="tight")
    plt.close()
    return fname

def summarize_inputs(inputs_dict):
    df = pd.DataFrame.from_dict(inputs_dict, orient="index", columns=["value"]).reset_index().rename(columns={"index":"feature"})
    df = df.sort_values("feature")
    fig_height = max(3, len(df) * 0.55)
    plt.figure(figsize=(10, fig_height))
    ax = sns.barplot(y=df["feature"], x=df["value"], palette="Blues_r")
    for i, v in enumerate(df["value"]):
        ax.text(v + (max(df["value"])*0.015), i, f"{v:,}", va="center", ha="left", fontsize=11, color="black", fontweight="bold")
    plt.title("Résumé clair des valeurs d’entrée", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Valeur", fontsize=12)
    plt.ylabel("Caractéristique", fontsize=12)
    plt.xlim(0, max(df["value"]) * 1.15)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "inputs_summary.png")
    plt.savefig(fname, dpi=240, bbox_inches="tight")
    plt.close()
    return fname

def plot_prediction_distribution(all_tree_preds, y_pred, lower, upper):
    plt.figure(figsize=(6, 3.5))
    plt.hist(all_tree_preds, bins=20, edgecolor="k", color="#4C72B0")
    plt.axvline(y_pred, color='black', linestyle='--', label=f'Pred {y_pred:,.0f}')
    plt.axvline(lower, color='red', linestyle=':', label=f'95% CI lower {lower:,.0f}')
    plt.axvline(upper, color='red', linestyle=':', label=f'95% CI upper {upper:,.0f}')
    plt.legend()
    plt.title("Distribution des prédictions des arbres")
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "prediction_with_ci.png")
    plt.savefig(fname, dpi=240, bbox_inches="tight")
    plt.close()
    return fname

# --- Prédiction ---
def predict_price(square_feet, bedrooms, bathrooms, age_years, lot_size, garage_spaces, neighborhood_score):
    X = pd.DataFrame([{
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "lot_size": lot_size,
        "garage_spaces": garage_spaces,
        "neighborhood_score": neighborhood_score
    }])
    X_scaled = scaler.transform(X)
    y_pred = float(model.predict(X_scaled)[0])
    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        all_tree_preds = np.array([est.predict(X_scaled)[0] for est in model.estimators_])
        std = float(all_tree_preds.std(ddof=1))
    else:
        all_tree_preds = np.array([y_pred])
        std = 0.0
    lower = y_pred - 1.96 * std
    upper = y_pred + 1.96 * std
    pred_plot_path = plot_prediction_distribution(all_tree_preds, y_pred, lower, upper)
    return round(y_pred, 2), (round(lower, 2), round(upper, 2)), pred_plot_path

# --- Interface Gradio ---
with gr.Blocks(css="""
    .gr-row { gap: 30px; }
    .gr-column { gap: 20px; }
    .gr-slider { width: 100%; }
    .gr-textbox { width: 100%; }
    .gr-image { object-fit: contain; max-height: 350px; border: 1px solid #ccc; border-radius: 8px; }
""") as demo:

    gr.Markdown("##  Prédicteur de prix immobiliers", elem_id="title")

    with gr.Row():
        # Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Entrées")
            sqft = gr.Slider(400, 4000, step=1, label="Surface (sqft)", value=1000)
            bedrooms = gr.Slider(1, 6, step=1, label="Chambres", value=3)
            bathrooms = gr.Slider(1, 4, step=1, label="Salles de bain", value=2)
            age = gr.Slider(0, 60, step=1, label="Âge (années)", value=10)
            lot = gr.Slider(200, 10000, step=1, label="Taille du lot (sqft)", value=1000)
            garage = gr.Slider(0, 4, step=1, label="Places garage", value=1)
            neigh = gr.Slider(0.0, 10.0, step=0.1, label="Score quartier", value=6.0)
            btn = gr.Button("Prédire", variant="primary")

        # Outputs
        with gr.Column(scale=1):
            gr.Markdown("### Résultats")
            out_price = gr.Textbox(label="Prix prédit", interactive=False)
            out_ci = gr.Textbox(label="Intervalle de confiance (95%)", interactive=False)
            with gr.Tabs():
                with gr.Tab("Importance des features"):
                    img_imp = gr.Image(label="Feature Importances")
                with gr.Tab("Résumé des inputs"):
                    img_inputs = gr.Image(label="Résumé des inputs")
                with gr.Tab("Distribution prédiction"):
                    img_pred = gr.Image(label="Prédiction et distribution des arbres")

    # Fonction de callback
    def on_predict(sqft_v, bed_v, bath_v, age_v, lot_v, gar_v, neigh_v):
        inputs = {
            "square_feet": sqft_v,
            "bedrooms": bed_v,
            "bathrooms": bath_v,
            "age_years": age_v,
            "lot_size": lot_v,
            "garage_spaces": gar_v,
            "neighborhood_score": neigh_v
        }
        imp_path = plot_feature_importance()
        inputs_path = summarize_inputs(inputs)
        pred_val, ci, pred_plot = predict_price(sqft_v, bed_v, bath_v, age_v, lot_v, gar_v, neigh_v)
        price_str = f"{pred_val:,.2f}"
        ci_str = f"({ci[0]:,.2f} — {ci[1]:,.2f})"
        return price_str, ci_str, imp_path, inputs_path, pred_plot

    btn.click(
        on_predict,
        inputs=[sqft, bedrooms, bathrooms, age, lot, garage, neigh],
        outputs=[out_price, out_ci, img_imp, img_inputs, img_pred]
    )

# Exécution
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
