#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modèle hybride de détection d'empreintes de mammifères pour WildLens
Intégrant les données visuelles et géographiques

Ce script implémente un modèle amélioré pour la classification d'empreintes
de 13 espèces de mammifères, en combinant l'analyse d'image avec les données
de distribution géographique pour améliorer la précision.

Auteur: Manus AI
Date: Mars 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
FINE_TUNING_EPOCHS = 30
LEARNING_RATE = 0.0001
FINE_TUNING_LEARNING_RATE = 0.00001
NUM_CLASSES = 13
RANDOM_SEED = 42

# Mapping des espèces
SPECIES_MAPPING = {
    0: 'Castor',
    1: 'Chat',
    2: 'Chien',
    3: 'Coyote',
    4: 'Écureuil',
    5: 'Lapin',
    6: 'Loup',
    7: 'Lynx',
    8: 'Ours',
    9: 'Puma',
    10: 'Rat',
    11: 'Raton Laveur',
    12: 'Renard'
}

# Régions géographiques (extraites du fichier CSV)
REGIONS = [
    'Asie', 'Moyen Orient', 'Amérique', 'Japon', 'Amérique Centrale',
    'Monde entier', 'Mexique', 'Afrique', 'Amérique du Sud', 'Australie',
    'Hémisphère nord', 'Russie', 'Eurasie', 'Asie Centrale', 'Europe du nord',
    'Moyen-Orient', 'Amérique du nord', 'Amérique du Nord', 'Afrique du Nord',
    'Europe', 'Amérique centrale', 'Asie occidentale'
]


def create_directories():
    """Crée les répertoires nécessaires pour le projet."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)


def load_geographic_data(csv_file):
    """
    Charge les données géographiques à partir du fichier CSV.

    Args:
        csv_file: Chemin vers le fichier CSV contenant les données géographiques

    Returns:
        DataFrame pandas contenant les données géographiques
    """
    print(f"Chargement des données géographiques depuis {csv_file}")

    # Chargement du fichier CSV
    geo_data = pd.read_csv(csv_file)

    # Nettoyage des données
    # Suppression de la première colonne sans nom si nécessaire
    if geo_data.columns[0] == '':
        geo_data = geo_data.drop(geo_data.columns[0], axis=1)

    print(f"Données géographiques chargées: {geo_data.shape[0]} espèces, {geo_data.shape[1]} colonnes")

    return geo_data


def create_species_region_matrix(geo_data):
    """
    Crée une matrice de présence des espèces par région.

    Args:
        geo_data: DataFrame pandas contenant les données géographiques

    Returns:
        Matrice numpy de forme (num_species, num_regions)
    """
    # Extraction des colonnes de régions (toutes les colonnes numériques après la colonne "Regions")
    region_cols = geo_data.select_dtypes(include=['int64', 'float64']).columns

    # Création de la matrice
    species_region_matrix = geo_data[region_cols].values

    return species_region_matrix


def preprocess_image(image):
    """
    Prétraite une image pour l'entrée du modèle.

    Args:
        image: Image d'entrée (BGR format from OpenCV)

    Returns:
        Image prétraitée et normalisée
    """
    # Redimensionnement à 224x224 pixels (format d'entrée de MobileNetV2)
    if isinstance(image, str):
        # Si l'entrée est un chemin de fichier
        image = cv2.imread(image)

    resized_img = cv2.resize(image, INPUT_SIZE)

    # Segmentation pour isoler l'empreinte du fond (optionnel selon la qualité des images)
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Application d'un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Forcer le type en uint8 après le flou
    blurred = np.uint8(blurred)

    # Seuillage adaptatif pour isoler l'empreinte
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Conversion BGR à RGB (TensorFlow utilise RGB)
    rgb_img = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

    # Normalisation des valeurs de pixels
    normalized_img = rgb_img / 255.0

    return normalized_img


def augmentation_generator():
    """
    Crée un générateur d'augmentation de données pour l'entraînement.

    Returns:
        Un générateur ImageDataGenerator configuré
    """
    return ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_image,
        validation_split=0.15  # 15% pour la validation
    )


def load_dataset(data_dir):
    """
    Charge le jeu de données à partir du répertoire spécifié.

    Args:
        data_dir: Chemin vers le répertoire contenant les images organisées par classe

    Returns:
        Générateurs pour l'entraînement, la validation et le test
    """
    print(f"Chargement des données depuis {data_dir}")

    # Création du générateur d'augmentation
    datagen = augmentation_generator()

    # Chargement des données d'entraînement et de validation
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=RANDOM_SEED
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=RANDOM_SEED
    )

    # Création d'un générateur de test séparé (sans augmentation)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        validation_split=0.15  # Utiliser la même proportion pour cohérence
    )

    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',  # Utiliser le même split que validation
        shuffle=False,
        seed=RANDOM_SEED
    )

    print(f"Classes trouvées: {train_generator.class_indices}")
    print(f"Nombre d'échantillons d'entraînement: {train_generator.samples}")
    print(f"Nombre d'échantillons de validation: {validation_generator.samples}")
    print(f"Nombre d'échantillons de test: {test_generator.samples}")

    return train_generator, validation_generator, test_generator


class HybridDataGenerator:
    """
    Générateur de données hybride qui combine les images d'empreintes avec les données géographiques.
    """

    def __init__(self, image_generator, species_region_matrix, region_indices=None):
        """
        Initialise le générateur hybride.

        Args:
            image_generator: Générateur d'images (flow_from_directory)
            species_region_matrix: Matrice de présence des espèces par région
            region_indices: Indices des régions à utiliser pour chaque batch (si None, utilise toutes les régions)
        """
        self.image_generator = image_generator
        self.species_region_matrix = species_region_matrix
        self.region_indices = region_indices
        self.num_regions = species_region_matrix.shape[1]
        self.batch_size = image_generator.batch_size
        self.n = 0  # Compteur pour __len__

    def __iter__(self):
        return self

    def __next__(self):
        """
        Génère un batch de données hybrides (images + données géographiques).

        Returns:
            Tuple ([images, region_data], labels)
        """
        # Obtenir le prochain batch d'images et d'étiquettes
        images, labels = next(self.image_generator)

        # Créer les données de région pour ce batch
        if self.region_indices is None:
            # Si aucun indice de région n'est fourni, utiliser des régions aléatoires
            region_indices = np.random.randint(0, self.num_regions, size=len(images))
        else:
            # Sinon, utiliser les indices fournis
            region_indices = self.region_indices

        # Créer un vecteur one-hot pour les régions
        region_data = np.zeros((len(images), self.num_regions))
        for i, idx in enumerate(region_indices):
            region_data[i, idx] = 1.0

        # Retourner les données hybrides
        return [images, region_data], labels

    def __len__(self):
        """Retourne le nombre de batches par epoch."""
        return len(self.image_generator)

    def reset(self):
        """Réinitialise le générateur d'images sous-jacent."""
        if hasattr(self.image_generator, 'reset'):
            self.image_generator.reset()
        # Si le générateur d'images n'a pas de méthode reset, on ne fait rien
        # car les générateurs Keras sont automatiquement réinitialisés à chaque époque

    @property
    def samples(self):
        """Retourne le nombre total d'échantillons."""
        return self.image_generator.samples

    @property
    def class_indices(self):
        """Retourne le dictionnaire des indices de classe."""
        return self.image_generator.class_indices

    @property
    def classes(self):
        """Retourne les indices de classe pour tous les échantillons."""
        return self.image_generator.classes

    @property
    def steps_per_epoch(self):
        """Retourne le nombre d'étapes par époque."""
        return len(self)


def build_hybrid_model(num_regions):
    """
    Construit le modèle hybride qui combine l'analyse d'image et les données géographiques.

    Args:
        num_regions: Nombre de régions géographiques

    Returns:
        Modèle Keras compilé
    """
    # Branche pour l'analyse d'image (MobileNetV2)
    image_input = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), name='image_input')
    base_model = MobileNetV2(input_tensor=image_input, include_top=False, weights='imagenet')

    # Geler les couches du modèle de base pour le transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    # Ajouter des couches personnalisées pour l'extraction de caractéristiques d'image
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    image_features = Dense(256, activation='relu', name='image_features')(x)

    # Branche pour les données géographiques
    geo_input = Input(shape=(num_regions,), name='geo_input')
    geo_features = Dense(64, activation='relu')(geo_input)
    geo_features = Dense(128, activation='relu')(geo_features)
    geo_features = Dense(256, activation='relu', name='geo_features')(geo_features)

    # Fusionner les deux branches
    combined = Concatenate()([image_features, geo_features])

    # Couches de classification finale
    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    # Créer le modèle avec les deux entrées et une sortie
    model = Model(inputs=[image_input, geo_input], outputs=output)

    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_hybrid_model(model, train_generator, validation_generator):
    """
    Entraîne le modèle hybride en deux phases.

    Args:
        model: Modèle hybride compilé
        train_generator: Générateur de données d'entraînement
        validation_generator: Générateur de données de validation

    Returns:
        Modèle entraîné et historiques d'entraînement
    """
    # Callbacks pour l'entraînement
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        'models/hybrid_model_phase1.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print("Phase 1: Entraînement avec les couches convolutives gelées...")

    # Calculer steps_per_epoch et validation_steps
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Phase 1: Entraînement avec les couches convolutives gelées
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    print("Phase 2: Fine-tuning des derniers blocs convolutifs...")

    # Chargement du meilleur modèle de la phase 1
    model = load_model('models/hybrid_model_phase1.h5')

    # Approche simplifiée pour le fine-tuning
    # Au lieu de chercher le modèle MobileNetV2, nous allons simplement
    # rendre entraînables les couches qui ont des noms contenant "mobilenetv2"
    # et qui sont situées dans la seconde moitié du modèle

    # Compter le nombre total de couches
    total_layers = len(model.layers)
    print(f"Nombre total de couches dans le modèle: {total_layers}")

    # Identifier les couches à rendre entraînables
    trainable_layers = 0
    for i, layer in enumerate(model.layers):
        # Ne rendre entraînables que les couches de la seconde moitié du modèle
        # qui contiennent "mobilenetv2" dans leur nom
        if i > total_layers // 2 and hasattr(layer, 'name') and 'mobilenetv2' in layer.name.lower():
            layer.trainable = True
            trainable_layers += 1
            print(f"Couche rendue entraînable: {layer.name}")

    # Si aucune couche MobileNetV2 n'a été trouvée, utiliser une approche alternative
    if trainable_layers == 0:
        print("Aucune couche MobileNetV2 trouvée. Utilisation d'une approche alternative...")

        # Rendre entraînables les 30% dernières couches du modèle
        start_index = int(total_layers * 0.7)
        for i, layer in enumerate(model.layers):
            if i >= start_index and hasattr(layer, 'trainable'):
                layer.trainable = True
                print(f"Couche rendue entraînable (approche alternative): {layer.name}")

    # Recompiler le modèle avec un taux d'apprentissage plus faible
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Mise à jour du checkpoint pour la phase 2
    checkpoint = ModelCheckpoint(
        'models/hybrid_model_phase2.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Phase 2: Fine-tuning
    history_fine_tuning = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=FINE_TUNING_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Chargement du meilleur modèle de la phase 2
    model = load_model('models/hybrid_model_phase2.h5')

    # Sauvegarde du modèle final
    model.save('models/hybrid_model_final.h5')

    return model, history, history_fine_tuning


def evaluate_hybrid_model(model, test_generator):
    """
    Évalue le modèle hybride sur le jeu de test.

    Args:
        model: Modèle hybride entraîné
        test_generator: Générateur de données de test

    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    print("Évaluation du modèle hybride...")

    # Calculer le nombre d'étapes pour l'évaluation
    steps = len(test_generator)

    # Évaluation du modèle
    loss, accuracy = model.evaluate(test_generator, steps=steps, verbose=1)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Prédictions sur le jeu de test
    y_pred = []
    y_true = []

    # Réinitialiser le générateur
    # Utiliser la méthode reset() que nous avons ajoutée à HybridDataGenerator
    test_generator.reset()

    # Collecter les prédictions et les vraies étiquettes
    for i in range(steps):
        try:
            [images, region_data], labels = next(test_generator)
            preds = model.predict([images, region_data], verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(labels, axis=1))
        except StopIteration:
            # En cas de StopIteration, réinitialiser le générateur et continuer
            test_generator.reset()
            [images, region_data], labels = next(test_generator)
            preds = model.predict([images, region_data], verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(labels, axis=1))

    # Calculer les métriques
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Rapport de classification détaillé
    class_names = [SPECIES_MAPPING[i] for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nRapport de classification:")
    print(report)

    # Sauvegarder les résultats
    results = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred
    }

    # Sauvegarder les résultats
    np.savez('results/hybrid_model_results.npz', **results)

    # Tracer et sauvegarder la matrice de confusion
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    plt.savefig('results/hybrid_confusion_matrix.png')

    return results


def plot_training_history(history, history_fine_tuning):
    """
    Trace les courbes d'apprentissage pour les deux phases d'entraînement.

    Args:
        history: Historique de la phase 1
        history_fine_tuning: Historique de la phase 2
    """
    # Combiner les historiques
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if history_fine_tuning is not None:
        acc += history_fine_tuning.history['accuracy']
        val_acc += history_fine_tuning.history['val_accuracy']
        loss += history_fine_tuning.history['loss']
        val_loss += history_fine_tuning.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Tracer l'accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Accuracy entraînement')
    plt.plot(epochs, val_acc, 'r', label='Accuracy validation')
    plt.axvline(x=len(history.history['accuracy']), color='g', linestyle='--',
                label='Début du fine-tuning')
    plt.title('Accuracy d\'entraînement et de validation')
    plt.legend()

    # Tracer la loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Loss entraînement')
    plt.plot(epochs, val_loss, 'r', label='Loss validation')
    plt.axvline(x=len(history.history['loss']), color='g', linestyle='--',
                label='Début du fine-tuning')
    plt.title('Loss d\'entraînement et de validation')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/hybrid_training_history.png')
    plt.close()


def compare_models(base_results, hybrid_results):
    """
    Compare les performances du modèle de base et du modèle hybride.

    Args:
        base_results: Résultats du modèle de base
        hybrid_results: Résultats du modèle hybride
    """
    # Métriques à comparer
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Créer un tableau de comparaison
    comparison = pd.DataFrame({
        'Métrique': metrics,
        'Modèle de base': [base_results[m] for m in metrics],
        'Modèle hybride': [hybrid_results[m] for m in metrics],
        'Amélioration (%)': [(hybrid_results[m] - base_results[m]) / base_results[m] * 100 for m in metrics]
    })

    print("\nComparaison des modèles:")
    print(comparison)

    # Sauvegarder la comparaison
    comparison.to_csv('results/model_comparison.csv', index=False)

    # Tracer un graphique de comparaison
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(metrics))

    plt.bar(index, [base_results[m] for m in metrics], bar_width, label='Modèle de base')
    plt.bar(index + bar_width, [hybrid_results[m] for m in metrics], bar_width, label='Modèle hybride')

    plt.xlabel('Métrique')
    plt.ylabel('Score')
    plt.title('Comparaison des performances des modèles')
    plt.xticks(index + bar_width / 2, metrics)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()


def predict_single_image(model, image_path, region_index):
    """
    Prédit l'espèce pour une seule image d'empreinte.

    Args:
        model: Modèle hybride entraîné
        image_path: Chemin vers l'image d'empreinte
        region_index: Indice de la région géographique

    Returns:
        Espèce prédite et score de confiance
    """
    # Prétraiter l'image
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Ajouter la dimension de batch

    # Créer les données de région
    region_data = np.zeros((1, len(REGIONS)))
    region_data[0, region_index] = 1.0

    # Prédire
    prediction = model.predict([img, region_data], verbose=0)

    # Obtenir l'indice de la classe prédite et le score de confiance
    predicted_class_index = np.argmax(prediction[0])
    confidence_score = prediction[0][predicted_class_index]

    # Obtenir le nom de l'espèce
    predicted_species = SPECIES_MAPPING[predicted_class_index]

    return predicted_species, confidence_score


def main():
    """Fonction principale."""
    # Créer les répertoires nécessaires
    create_directories()

    # Charger les données géographiques
    geo_data = load_geographic_data('infos_especes_lieu.csv')

    # Créer la matrice de présence des espèces par région
    species_region_matrix = create_species_region_matrix(geo_data)
    num_regions = species_region_matrix.shape[1]

    # Charger le jeu de données
    train_generator, validation_generator, test_generator = load_dataset('Animaux_clean')

    # Créer les générateurs hybrides
    hybrid_train_generator = HybridDataGenerator(train_generator, species_region_matrix)
    hybrid_validation_generator = HybridDataGenerator(validation_generator, species_region_matrix)
    hybrid_test_generator = HybridDataGenerator(test_generator, species_region_matrix)

    # Construction du modèle hybride
    hybrid_model = build_hybrid_model(num_regions)

    # Entraînement du modèle hybride
    hybrid_model, history, history_fine_tuning = train_hybrid_model(
        hybrid_model, hybrid_train_generator, hybrid_validation_generator
    )

    # Évaluation du modèle hybride
    hybrid_results = evaluate_hybrid_model(hybrid_model, hybrid_test_generator)

    # Tracé des courbes d'apprentissage
    plot_training_history(history, history_fine_tuning)

    # Chargement des résultats du modèle de base (si disponibles)
    base_results_file = 'results/base_model_results.npz'
    if os.path.exists(base_results_file):
        base_results = np.load(base_results_file, allow_pickle=True)
        base_results = dict(base_results)

        # Comparaison des modèles
        compare_models(base_results, hybrid_results)

    print("Entraînement et évaluation terminés avec succès!")
    print("Modèle hybride sauvegardé dans 'models/hybrid_model_final.h5'")
    print("Résultats sauvegardés dans le répertoire 'results/'")


if __name__ == "__main__":
    main()
