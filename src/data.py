
from pathlib import Path

import pandas as pd

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_manifest(data_dir: str | Path):
    root = Path(data_dir)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset introuvable : {root.resolve()}"
        )

    records = []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            records.append({
                "path": str(path.resolve()),
                "label": class_dir.name,
                "extension": path.suffix.lower(),
            })

    manifest = pd.DataFrame(
        records,
        columns=["path", "label", "extension"],
    )

    if manifest.empty:
        raise ValueError("Aucune image valide trouvée.")

    if manifest["label"].nunique() < 2:
        raise ValueError(
            "Au moins deux classes sont nécessaires."
        )

    return manifest


if __name__ == "__main__":
    manifest = build_manifest("data/brain_mri")

    print(manifest.head())
    print("\nNombre d'images :", len(manifest))
    print("\nDistribution :")
    print(manifest["label"].value_counts())

    manifest.to_csv(
        "artifacts/manifest.csv",
        index=False,
    )