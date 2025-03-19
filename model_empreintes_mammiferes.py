#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modèle de détection d'empreintes de mammifères pour WildLens
Basé sur MobileNetV2 avec transfer learning

Ce script implémente un modèle complet pour la classification d'empreintes
de 13 espèces de mammifères, incluant le prétraitement des données,
l'augmentation de données, la construction du modèle, l'entraînement
et l'évaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
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
    6: 'Rat',
    7: 'Raton Laveur',
    8: 'Renard',
    9: 'Souris',
    10: 'Belette',
    11: 'Marmotte',
    12: 'Moufette'
}

def create_directories():
    """Crée les répertoires nécessaires pour le projet."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)


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
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

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

def build_model():
    """
    Construit le modèle MobileNetV2 avec transfer learning.
    
    Returns:
        Modèle compilé
    """
    print("Construction du modèle MobileNetV2 avec transfer learning...")
    
    # Chargement du modèle MobileNetV2 pré-entraîné sans les couches fully-connected
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3)
    )
    
    # Gel des couches convolutives
    for layer in base_model.layers:
        layer.trainable = False
    
    # Construction du modèle complet
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')  # 13 classes pour les espèces de mammifères
    ])
    
    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    return model

def train_model(model, train_generator, validation_generator):
    """
    Entraîne le modèle en deux phases: feature extraction puis fine-tuning.
    
    Args:
        model: Modèle à entraîner
        train_generator: Générateur pour les données d'entraînement
        validation_generator: Générateur pour les données de validation
        
    Returns:
        Modèle entraîné et historiques d'entraînement
    """
    print("Phase 1: Entraînement des couches fully-connected uniquement...")
    
    # Définition des callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'models/model_phase1.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Phase 1: Entraînement des couches fully-connected uniquement
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
    
    print("Phase 2: Fine-tuning des derniers blocs convolutifs...")
    
    # Chargement du meilleur modèle de la phase 1
    model = load_model('models/model_phase1.h5')
    
    # Dégel des 3 derniers blocs de MobileNetV2 (environ 23 couches)
    base_model = model.layers[0]
    for layer in base_model.layers[-23:]:
        layer.trainable = True
    
    # Recompilation avec un taux d'apprentissage plus faible
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Mise à jour du checkpoint pour la phase 2
    checkpoint = ModelCheckpoint(
        'models/model_phase2.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Poursuite de l'entraînement
    history_fine_tuning = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=FINE_TUNING_EPOCHS,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
    
    # Chargement du meilleur modèle de la phase 2
    model = load_model('models/model_phase2.h5')
    
    # Sauvegarde du modèle final
    model.save('models/model_final.h5')
    
    return model, history, history_fine_tuning

def evaluate_model(model, test_generator):
    """
    Évalue le modèle sur les données de test.
    
    Args:
        model: Modèle entraîné
        test_generator: Générateur pour les données de test
        
    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    print("Évaluation du modèle sur les données de test...")
    
    # Réinitialisation du générateur de test
    test_generator.reset()
    
    # Évaluation du modèle sur les données de test
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Prédictions sur les données de test
    test_generator.reset()
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Récupération des vraies étiquettes
    y_true = test_generator.classes[:len(y_pred)]
    
    # Calcul de la matrice de confusion
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calcul des métriques par classe
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Rapport de classification
    class_report = classification_report(y_true, y_pred, target_names=list(SPECIES_MAPPING.values()))
    print("Rapport de classification:")
    print(class_report)
    
    # Sauvegarde des résultats
    np.save('results/confusion_matrix.npy', conf_matrix)
    np.save('results/precision.npy', precision)
    np.save('results/recall.npy', recall)
    np.save('results/f1_score.npy', f1)
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.colorbar()
    tick_marks = np.arange(len(SPECIES_MAPPING))
    plt.xticks(tick_marks, SPECIES_MAPPING.values(), rotation=90)
    plt.yticks(tick_marks, SPECIES_MAPPING.values())
    plt.tight_layout()
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig('results/confusion_matrix.png')
    
    return {
        'accuracy': test_accuracy,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report
    }

def plot_training_history(history, history_fine_tuning):
    """
    Trace les courbes d'apprentissage pour les deux phases d'entraînement.
    
    Args:
        history: Historique de la phase 1
        history_fine_tuning: Historique de la phase 2
    """
    # Combinaison des historiques
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    acc += history_fine_tuning.history['accuracy']
    val_acc += history_fine_tuning.history['val_accuracy']
    loss += history_fine_tuning.history['loss']
    val_loss += history_fine_tuning.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # Tracé de l'accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Accuracy entraînement')
    plt.plot(epochs, val_acc, 'r', label='Accuracy validation')
    plt.axvline(x=len(history.history['accuracy']), color='g', linestyle='--', 
                label='Début du fine-tuning')
    plt.title('Accuracy d\'entraînement et de validation')
    plt.legend()
    
    # Tracé de la loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Loss entraînement')
    plt.plot(epochs, val_loss, 'r', label='Loss validation')
    plt.axvline(x=len(history.history['loss']), color='g', linestyle='--', 
                label='Début du fine-tuning')
    plt.title('Loss d\'entraînement et de validation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

def predict_single_image(model, image_path):
    """
    Prédit l'espèce à partir d'une seule image d'empreinte.
    
    Args:
        model: Modèle entraîné
        image_path: Chemin vers l'image à prédire
        
    Returns:
        Dictionnaire contenant les résultats de prédiction
    """
    # Chargement et prétraitement de l'image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Impossible de charger l'image: {image_path}"}
    
    preprocessed_img = preprocess_image(image)
    
    # Préparation pour l'entrée du modèle
    input_img = np.expand_dims(preprocessed_img, axis=0)
    
    # Prédiction
    predictions = model.predict(input_img)
    
    # Récupération de la classe avec la plus haute probabilité
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Récupération du nom de l'espèce
    predicted_species = SPECIES_MAPPING[predicted_class_index]
    
    # Top 3 prédictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_predictions = [
        {'species': SPECIES_MAPPING[i], 'confidence': float(predictions[0][i])}
        for i in top_indices
    ]
    
    return {
        'species': predicted_species,
        'confidence': float(confidence),
        'top_predictions': top_predictions
    }

def main():
    """Fonction principale."""
    # Création des répertoires nécessaires
    create_directories()
    
    # Définition du chemin vers les données
    # Remplacer par le chemin réel vers votre jeu de données
    data_dir = "Animaux_clean/"
    
    # Vérification si le répertoire de données existe
    if not os.path.exists(data_dir):
        print(f"ERREUR: Le répertoire de données {data_dir} n'existe pas.")
        print("Veuillez créer ce répertoire avec la structure suivante:")
        print(f"{data_dir}/")
        print("├── castor/")
        print("├── chat/")
        print("├── chien/")
        print("└── ... (autres espèces)")
        return
    
    # Chargement des données
    train_generator, validation_generator, test_generator = load_dataset(data_dir)
    
    # Construction du modèle
    model = build_model()
    
    # Entraînement du modèle
    model, history, history_fine_tuning = train_model(model, train_generator, validation_generator)
    
    # Évaluation du modèle
    evaluation_results = evaluate_model(model, test_generator)
    
    # Tracé des courbes d'apprentissage
    plot_training_history(history, history_fine_tuning)
    
    print("Entraînement et évaluation terminés.")
    print(f"Modèle final sauvegardé dans 'models/model_final.h5'")
    print(f"Résultats sauvegardés dans le répertoire 'results/'")
    
    # Exemple de prédiction sur une image de test
    # Remplacer par le chemin vers une image de test
    test_image = "Animaux_clean/Lapin/Black-tailed-Jackrabbit-Tracks-3.jpg"
    if os.path.exists(test_image):
        prediction = predict_single_image(model, test_image)
        print("\nPrédiction pour l'image de test:")
        print(f"Espèce: {prediction['species']}")
        print(f"Confiance: {prediction['confidence']:.2f}")
        print("Top 3 prédictions:")
        for pred in prediction['top_predictions']:
            print(f"  {pred['species']}: {pred['confidence']:.2f}")

if __name__ == "__main__":
    # Définition des seeds pour reproductibilité
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Exécution de la fonction principale
    main()
