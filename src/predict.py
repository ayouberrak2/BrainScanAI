
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.preprocessing import load_image


# Racine du projet BrainScan
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chemin du modèle converti en .h5
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "best_model.h5"
)


def load_saved_model(model_path=MODEL_PATH):
    """Charge le modèle entraîné sauvegardé en .h5."""
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path.resolve()}"
        )

    model = keras.models.load_model(model_path)

    print(f"Modèle chargé : {model_path.resolve()}")

    return model


def predict_image(path, model, class_names):
    """Prédit la classe d'une image IRM avec le CNN entraîné."""
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image introuvable : {image_path.resolve()}"
        )

    # Prétraitement identique à celui de l'entraînement
    image = load_image(
        str(image_path),
        image_size=(224, 224),
    )

    # Ajouter la dimension du batch : (1, 224, 224, 3)
    image_batch = np.expand_dims(image, axis=0)

    # Prédiction
    probabilities = model.predict(
        image_batch,
        verbose=0,
    )[0]

    if len(class_names) != len(probabilities):
        raise ValueError(
            "Le nombre de classes ne correspond pas "
            "aux sorties du modèle."
        )

    predicted_id = int(np.argmax(probabilities))

    return {
        "image": str(image_path),
        "predicted_class": class_names[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": probabilities.tolist(),
    }
