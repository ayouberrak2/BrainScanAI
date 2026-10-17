
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow import keras


def load_best_model(
    model_path: str | Path = "artifacts/models/best_model.keras",
):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    return keras.models.load_model(model_path)


def evaluate_model(model, test_ds):
    loss, accuracy = model.evaluate(test_ds, verbose=1)

    print(f"Test loss     : {loss:.4f}")
    print(f"Test accuracy : {accuracy:.2%}")

    return {"test_loss": loss, "test_accuracy": accuracy}


def plot_training_history(history):
    metrics = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss", "val_loss", "Loss"),
    ]

    for train_key, val_key, title in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(history[train_key], label="Train")
        plt.plot(history[val_key], label="Validation")
        plt.title(f"BrainScan AI - {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def get_predictions(model, test_ds):
    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Le nombre de labels et de prédictions diffère."
        )

    return y_true, y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))

    cm = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
    )
    ax.set_title("BrainScan AI - Matrice de confusion")
    fig.tight_layout()
    plt.show()

    return cm


def print_classification_report(
    y_true, y_pred, class_names
):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(report)
    return report

