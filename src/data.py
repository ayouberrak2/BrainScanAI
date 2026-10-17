
from pathlib import Path
import hashlib
import matplotlib.pyplot as plt


import pandas as pd


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest(data_dir: str | Path) -> pd.DataFrame:
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
                "sha256": sha256_file(path),
            })

    manifest = pd.DataFrame(
        records,
        columns=["path", "label", "extension", "sha256"],
    )

    if manifest.empty:
        raise ValueError("aucune image valide trouvee.")

    return manifest


def analyze_classes(manifest: pd.DataFrame) -> pd.Series:
    if manifest.empty:
        raise ValueError("Le dataset ne contient aucune image.")

    class_counts = manifest["label"].value_counts().sort_index()


    class_counts.plot(kind="bar", title="Distribution des classes")
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return class_counts







# if __name__ == "__main__":

#     manifest = build_manifest("data")
#     analyze_classes(manifest)

#     