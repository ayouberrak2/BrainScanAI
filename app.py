
import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import load_saved_model, predict_image

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES_PATH = PROJECT_ROOT / "artifacts" / "class" / "class_names.json"

st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="wide",
)

# Style de l'interface
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #4F8BF9;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9AA4B2;
        font-size: 17px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """Charge le CNN une seule fois."""
    return load_saved_model()


@st.cache_data
def get_class_names():
    """Charge les classes dans l'ordre utilisé à l'entraînement."""
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier des classes introuvable : {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# En-tête
st.markdown(
    '<p class="main-title">🧠 BrainScan AI</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Classification expérimentale des IRM cérébrales par CNN</p>',
    unsafe_allow_html=True,
)

st.info(
    "Prototype éducatif : les prédictions ne constituent pas un diagnostic "
    "médical et doivent être interprétées par un professionnel de santé."
)

# Chargement des ressources
try:
    model = get_model()
    class_names = get_class_names()
except Exception as error:
    st.error(f"Impossible de charger le modèle ou les classes : {error}")
    st.stop()

# Zone de chargement
st.subheader("1. Importer une image IRM")

uploaded_file = st.file_uploader(
    "Choisis une image au format JPG, JPEG ou PNG",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Image importée")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with right:
            st.subheader("Analyse du modèle")
            st.write("Clique sur le bouton pour lancer la prédiction.")

            if st.button("🔍 Analyser l'image", type="primary"):
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    image.save(temp_path, format="PNG")

                try:
                    with st.spinner("Analyse en cours..."):
                        result = predict_image(
                            path=str(temp_path),
                            model=model,
                            class_names=class_names,
                        )

                    st.success("Prédiction terminée")

                    st.metric(
                        label="Classe prédite",
                        value=result["predicted_class"],
                    )

                    confidence = result["confidence"]
                    st.metric(
                        label="Score du modèle",
                        value=f"{confidence:.2%}",
                    )

                    st.progress(max(0.0, min(1.0, confidence)))

                    st.caption(
                        "Ce score représente la probabilité estimée par le "
                        "modèle, pas la certitude d'un diagnostic."
                    )

                    st.subheader("Probabilités par classe")

                    for name, probability in zip(
                        class_names, result["probabilities"]
                    ):
                        st.write(f"**{name}** — {probability:.2%}")
                        st.progress(
                            max(0.0, min(1.0, float(probability)))
                        )

                except Exception as error:
                    st.error(f"Erreur pendant la prédiction : {error}")

                finally:
                    temp_path.unlink(missing_ok=True)

    except Exception as error:
        st.error(f"Impossible de lire cette image : {error}")
else:
    st.warning("Importe une image IRM pour commencer.")

st.divider()
st.caption("BrainScan AI | Prototype de classification par Deep Learning")
