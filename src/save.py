
from pathlib import Path
from tensorflow import keras


def convert_keras_to_h5(
    input_path="artifacts/models/best_model.keras",
    output_path="artifacts/models/best_model.h5",
):
    """Convertit un modèle .keras en .h5 sans réentraînement."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Modèle .keras introuvable : {input_path.resolve()}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Charger le modèle déjà entraîné
    model = keras.models.load_model(input_path)

    # Sauvegarder au format HDF5
    model.save(str(output_path))

    print(f"Modèle converti avec succès : {output_path.resolve()}")

    return output_path
