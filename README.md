
# 🧠 BrainScan AI
### Brain MRI Classification Using Convolutional Neural Networks

## 1. Description du projet

**BrainScan AI** est un projet de Deep Learning qui vise à classifier des images IRM (Imagerie par Résonance Magnétique) cérébrales à l'aide d'un réseau de neurones convolutif (CNN — Convolutional Neural Network).

Le projet utilise Python, TensorFlow/Keras et des techniques de traitement d'images pour apprendre à reconnaître différentes catégories d'IRM cérébrales.

L'objectif est de développer un prototype capable de recevoir une image IRM, de la prétraiter, puis de prédire la classe correspondante.

### Objectifs principaux

- Préparer et organiser un dataset d'images IRM cérébrales.
- Nettoyer et vérifier les données avant l'entraînement.
- Prétraiter les images pour les rendre compatibles avec le CNN.
- Développer une architecture CNN pour la classification d'images.
- Entraîner et valider le modèle.
- Évaluer les performances à l'aide de métriques de classification.
- Sauvegarder le meilleur modèle entraîné.
- Tester les prédictions sur de nouvelles images.
- Développer une interface web avec Streamlit pour faciliter les tests.

### Classes de classification

Selon les classes présentes dans le dataset, le modèle peut classifier les images dans les catégories suivantes :

| Classe | Description |
|---|---|
| `glioma` | Catégorie associée aux gliomes |
| `meningioma` | Catégorie associée aux méningiomes |
| `notumor` | Image classée dans la catégorie sans tumeur |
| `pituitary` | Catégorie associée aux tumeurs hypophysaires |

> Important : les catégories et leur ordre doivent correspondre exactement aux labels utilisés pendant l'entraînement. La liste ci-dessus est indicative et doit être vérifiée par rapport au dataset réel.

### Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Langage principal |
| TensorFlow / Keras | Construction et entraînement du CNN |
| NumPy | Manipulation des tableaux numériques |
| Pandas | Organisation et analyse des métadonnées |
| Pillow (PIL) | Lecture et manipulation des images |
| Matplotlib | Visualisation des résultats |
| Scikit-learn | Métriques et évaluation du modèle |
| Streamlit | Création de l'interface web |
| Git / GitHub | Gestion de versions et collaboration |
| Jupyter Notebook | Expérimentation et analyse |

---

## 2. Architecture du projet

Le projet est organisé en plusieurs modules afin de séparer la préparation des données, la construction du modèle, l'entraînement, l'évaluation et les prédictions.

### Structure des fichiers

```text
BrainScan/
│
├── artifacts/
│   └── models/
│       ├── best_model.h5
│       ├── best_model.keras
│       └── class_names.json
│
├── data/
│   ├── raw/
│   │   └── Images originales du dataset
│   └── processed/
│       └── Métadonnées et données préparées
│
├── notebooks/
│   └── training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── save.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> Cette structure représente l'organisation cible du projet. Certains dossiers ou fichiers peuvent être absents selon l'état actuel du dépôt. Adapte l'arborescence aux fichiers réellement présents.

### Rôle des dossiers

#### `artifacts/models/`

Contient les modèles entraînés et les informations nécessaires à l'inférence.

- `best_model.keras` : modèle sauvegardé au format Keras natif.
- `best_model.h5` : modèle sauvegardé au format HDF5.
- `class_names.json` : noms des classes dans l'ordre exact utilisé par le modèle.

#### `data/`

Contient les données utilisées dans le projet.

- `raw/` : images originales.
- `processed/` : métadonnées ou données préparées.

Les images médicales utilisées doivent provenir de sources autorisées et être traitées dans le respect de la confidentialité.

#### `notebooks/`

Contient les notebooks Jupyter utilisés pour explorer les données, entraîner le modèle, analyser les résultats et expérimenter différentes architectures.

#### `src/`

Contient le code Python réutilisable du projet.

| Fichier | Responsabilité |
|---|---|
| `config.py` | Paramètres et chemins du projet |
| `data.py` | Découverte et chargement des données |
| `preprocessing.py` | Prétraitement et préparation des images |
| `model.py` | Construction de l'architecture CNN |
| `train.py` | Entraînement et sauvegarde du meilleur modèle |
| `evaluate.py` | Évaluation et visualisation des performances |
| `predict.py` | Chargement du modèle et prédiction d'une image |
| `save.py` | Fonctions complémentaires de sauvegarde |

#### `app.py`

Contient l'application Streamlit permettant à l'utilisateur de téléverser une image IRM et de consulter les résultats de la prédiction.

#### `requirements.txt`

Liste les dépendances Python nécessaires à l'exécution du projet.

---

## 3. Prérequis

Avant d'exécuter le projet, vérifie que les outils suivants sont installés :

- Python 3.11 ou une version compatible avec les versions de TensorFlow et des autres dépendances du projet.
- pip, pour installer les bibliothèques Python.
- Git, pour gérer le code source.
- Visual Studio Code ou un autre éditeur Python.
- Un dataset d'images IRM organisé selon les classes à prédire.

Un GPU compatible peut accélérer l'entraînement, mais il n'est pas obligatoire pour effectuer des tests sur un modèle déjà entraîné.

Vérifier les installations :

```bash
python --version
python -m pip --version
git --version
```

---

## 4. Installation du projet

### Étape 1 — Récupérer le projet

Si le dépôt est hébergé sur GitHub :

```bash
git clone <URL_DU_DEPOT>
cd BrainScan
```

Remplace `<URL_DU_DEPOT>` par l'URL réelle de ton dépôt.

Si le projet existe déjà sur ton ordinateur, ouvre simplement son dossier dans VS Code.

### Étape 2 — Créer un environnement virtuel

Sous Windows PowerShell :

```powershell
python -m venv .venv
```

### Étape 3 — Activer l'environnement virtuel

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloque l'activation à cause de la politique d'exécution, tu peux utiliser l'invite de commandes Windows :

```cmd
.venv\Scripts\activate.bat
```

### Étape 4 — Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si `requirements.txt` n'est pas encore complet, installe les dépendances nécessaires :

```bash
python -m pip install tensorflow numpy pandas matplotlib scikit-learn pillow streamlit
```

Pour reproduire les résultats, conserve les versions compatibles de TensorFlow, Keras et des autres bibliothèques dans `requirements.txt`.

---

## 5. Préparation des données

Avant l'entraînement, les images doivent être organisées et associées à des labels corrects.

Exemple d'organisation possible :

```text
data/raw/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

Chaque dossier contient les images appartenant à la classe correspondante.

Les étapes de préparation peuvent comprendre :

1. Vérification des extensions et de la lisibilité des images.
2. Construction d'un manifeste contenant les chemins et les labels.
3. Suppression ou identification des doublons.
4. Conversion des images au format attendu.
5. Redimensionnement vers la taille utilisée par le CNN, par exemple `224 × 224`.
6. Normalisation des pixels selon la méthode définie dans le pipeline.
7. Séparation des données en ensembles d'entraînement, de validation et de test.

Le découpage doit éviter que des images identiques ou quasi identiques se retrouvent dans plusieurs ensembles.

La séparation de test doit être conservée pour l'évaluation finale et ne doit pas servir à ajuster le modèle.

---

## 6. Entraîner le modèle CNN

L'entraînement est réalisé à partir des modules du dossier `src/` ou du notebook d'entraînement.

Exemple d'utilisation depuis Python :

```python
from src.train import train_model

num_classes = len(class_names)

best_model, history, duration = train_model(
    train_ds=train_ds,
    val_ds=val_ds,
    num_classes=num_classes,
    epochs=30,
)

print(f"Durée de l'entraînement : {duration:.2f} secondes")
```

Les variables `class_names`, `train_ds` et `val_ds` doivent avoir été définies au préalable pendant la préparation des données.

Selon l'implémentation de `train.py`, le processus peut inclure :

- La sauvegarde du meilleur modèle.
- L'arrêt anticipé avec `EarlyStopping`.
- La réduction automatique du taux d'apprentissage.
- L'enregistrement de l'historique d'entraînement.

Le modèle sauvegardé est ensuite utilisé pour les prédictions et l'évaluation.

> Le code ci-dessus est un exemple d'utilisation de l'API du projet ; il suppose que les fonctions et variables correspondantes ont déjà été préparées.

---

## 7. Évaluer le modèle

Après l'entraînement, le modèle doit être évalué sur l'ensemble de test.

Exemple :

```python
from src.evaluate import (
    load_best_model,
    evaluate_model,
    get_predictions,
    plot_confusion_matrix,
    print_classification_report,
)

best_model = load_best_model()

test_metrics = evaluate_model(
    best_model,
    test_ds,
)

y_true, y_pred, y_prob = get_predictions(
    best_model,
    test_ds,
)

plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
)

report = print_classification_report(
    y_true,
    y_pred,
    class_names,
)
```

Les fonctions et les noms de paramètres doivent correspondre à leur implémentation réelle dans `src/evaluate.py`.

### Métriques à examiner

- **Accuracy** : proportion totale de prédictions correctes.
- **Precision** : proportion de prédictions positives correctes pour une classe.
- **Recall** : proportion des exemples d'une classe correctement retrouvés.
- **F1-score** : compromis entre précision et rappel.
- **Matrice de confusion** : répartition des prédictions correctes et incorrectes.

Dans un projet médical expérimental, l'accuracy seule ne suffit pas. Il faut également étudier les erreurs par classe et les limites du dataset.

---

## 8. Tester une prédiction individuelle

Une image peut être testée à l'aide du module `src/predict.py`.

Exemple :

```python
from src.predict import load_saved_model, predict_image

model = load_saved_model()

result = predict_image(
    path="chemin/vers/une_image.jpg",
    model=model,
    class_names=class_names,
)

print("Classe prédite :", result["predicted_class"])
print("Confiance estimée :", f'{result["confidence"]:.2%}')
print("Probabilités :", result["probabilities"])
```

Remplace `chemin/vers/une_image.jpg` par le chemin d'une image existante.

Les noms de classes doivent respecter exactement le mapping utilisé pendant l'entraînement.

---

## 9. Lancer l'application Streamlit

L'interface permet de tester le modèle sans écrire de code pour chaque image.

### Étape 1 — Vérifier les fichiers requis

Assure-toi que les fichiers suivants existent :

```text
app.py
artifacts/models/best_model.h5
artifacts/models/class_names.json
src/predict.py
src/preprocessing.py
```

Si ton application utilise le modèle `.keras` plutôt que `.h5`, adapte le chemin dans `src/predict.py`.

### Étape 2 — Démarrer l'application

Depuis la racine du projet :

```bash
streamlit run app.py
```

### Étape 3 — Ouvrir l'interface

Streamlit affiche normalement une adresse locale, par exemple :

```text
http://localhost:8501
```

Ouvre cette adresse dans ton navigateur.

### Étape 4 — Effectuer une prédiction

1. Sélectionne une image au format JPG, JPEG ou PNG.
2. Vérifie que l'image affichée correspond au fichier choisi.
3. Clique sur le bouton d'analyse.
4. Consulte la classe prédite.
5. Examine les scores affichés pour les différentes classes.

Les scores représentent les sorties du modèle et ne doivent pas être interprétés comme une certitude clinique.

---

## 10. Problèmes fréquents

### Erreur : modèle introuvable

Vérifie que le modèle existe bien dans `artifacts/models/` et que le chemin configuré dans `src/predict.py` correspond au fichier réel.

### Erreur : `NameError: name 'class_names' is not defined`

La variable n'a pas été définie dans la session Python en cours. Exécute de nouveau les cellules de préparation des labels ou charge le mapping sauvegardé dans `class_names.json`.

### Erreur : nombre de classes incorrect

Le nombre de noms de classes doit correspondre au nombre de sorties du modèle. Vérifie aussi l'ordre des classes utilisé pendant l'entraînement.

### Erreur : module introuvable

Active l'environnement virtuel et installe les dépendances :

```bash
python -m pip install -r requirements.txt
```

Lance également Streamlit depuis la racine du projet afin que les imports `src.*` puissent être résolus.

### L'application ne démarre pas

Vérifie que Streamlit est installé dans le même environnement Python que celui utilisé pour lancer l'application :

```bash
python -m streamlit run app.py
```

### Résultats incohérents

Vérifie la cohérence entre le prétraitement de l'entraînement et celui des prédictions : taille des images, canaux de couleur, normalisation et ordre des classes.

---

## 11. Bonnes pratiques

- Utiliser un environnement virtuel pour isoler les dépendances.
- Garder les fonctions de préparation, d'entraînement, d'évaluation et de prédiction séparées.
- Fixer les graines aléatoires lorsque cela est approprié.
- Conserver les métriques et les historiques d'entraînement.
- Éviter de versionner les environnements virtuels, les caches Python et les gros modèles si la stratégie de stockage du projet ne le prévoit pas.
- Utiliser un `.gitignore` adapté.
- Documenter les versions des bibliothèques et la provenance du dataset.
- Respecter la confidentialité et les licences des données utilisées.

---

## 12. Limites et responsabilité médicale

BrainScan AI est un projet d'apprentissage et d'expérimentation en Deep Learning.

Une classification correcte sur un dataset de test ne prouve pas que le modèle est fiable dans un environnement clinique réel. Les performances peuvent varier selon la qualité des images, les équipements d'imagerie, les populations étudiées et la provenance des données.

Le système ne remplace pas l'expertise d'un radiologue ou d'un médecin. Il ne doit pas être utilisé seul pour établir un diagnostic, décider d'un traitement ou exclure une pathologie.

Toute utilisation clinique nécessiterait notamment une validation indépendante, des tests sur des données représentatives, une analyse des erreurs et le respect des exigences réglementaires applicables.

---

## 13. Auteur et contribution

**Projet :** BrainScan AI  
**Domaine :** Intelligence artificielle, Deep Learning et analyse d'images médicales  
**Langage principal :** Python

Les contributions peuvent inclure l'amélioration du pipeline de données, l'expérimentation de nouvelles architectures CNN, l'évaluation des modèles et l'amélioration de l'interface utilisateur.

Avant de proposer une contribution, vérifie la structure du dépôt et les dépendances du projet.

---

## Licence

Ajoute une licence au dépôt si tu souhaites autoriser explicitement la réutilisation, la modification ou la distribution du code. Vérifie séparément les conditions de licence du dataset utilisé.
