
import time
from pathlib import Path

from tensorflow import keras

from src.model import build_cnn


def train_model(
    train_ds,
    val_ds,
    num_classes: int,
    output_dir: str | Path = "artifacts/models",
    epochs: int = 30,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_model.keras"

    model = build_cnn(num_classes=num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(output_dir / "training_log.csv")
        ),
    ]

    start = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    duration = time.perf_counter() - start

    best_model = keras.models.load_model(model_path)

    return best_model, history.history, duration


if __name__ == "__main__":
    print(
        "Construis les datasets train_ds et val_ds "
        "dans le pipeline principal avant d'appeler train_model()."
    )
