from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd




def load_image( path, image_size = (224, 224) ):

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Image illisible : {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        image_size,
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    return image


def split_dataset( manifest, test_size = 0.15, val_size = 0.15):

    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("Les proportions doivent être comprises entre 0 et 1.")

    if test_size + val_size >= 1:
        raise ValueError("La somme de val_size et test_size doit être < 1.")

    if "label" not in manifest.columns:
        raise ValueError("Le manifest doit contenir une colonne 'label'.")

    if manifest["label"].isna().any():
        raise ValueError("La colonne 'label' contient des valeurs manquantes.")

    train_df, temp_df = train_test_split(
        manifest,
        test_size=test_size + val_size,
        random_state=42 ,
        stratify=manifest["label"],
    )

    relative_test_size = test_size / (test_size + val_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=42,
        stratify=temp_df["label"],
    )

    return train_df, val_df, test_df


def encode_labels(train_df, val_df, test_df):

    class_names = sorted(train_df["label"].unique())
    label_to_id = {
        label: index
        for index, label in enumerate(class_names)
    }

    for name, df in [
        ("validation", val_df),
        ("test", test_df),
    ]:
        unknown = set(df["label"].unique()) - set(class_names)
        if unknown:
            raise ValueError(
                f"Classes inconnues dans {name} : {unknown}"
            )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for df in (train_df, val_df, test_df):
        df["label_id"] = df["label"].map(label_to_id)

    return train_df, val_df, test_df, class_names




def create_tf_dataset( df, batch_size= 32, shuffle= False, image_size= (224, 224) ):
    if df.empty:
        raise ValueError("Le DataFrame ne contient aucune image.")

    paths = df["path"].astype(str).to_numpy()
    labels = df["label_id"].to_numpy(dtype="int32")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_sample(path, label):
        image = tf.numpy_function(
            func=lambda p: load_image(
                p.decode("utf-8"),
                image_size=image_size,
            ),
            inp=[path],
            Tout=tf.float32,
        )

        image.set_shape((image_size[1], image_size[0], 3))
        label = tf.cast(label, tf.int32)

        return image, label

    dataset = dataset.map(
        load_sample,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
