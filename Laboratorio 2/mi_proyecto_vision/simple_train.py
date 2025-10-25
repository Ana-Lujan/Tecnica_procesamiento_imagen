"""
Entrenamiento simple de modelo CNN para clasificación de residuos
Versión simplificada para evitar problemas de memoria
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Configuración
DATASET_PATH = './'
IMAGE_SIZE = (100, 100)
BATCH_SIZE = 8
EPOCHS = 10

def create_data_generators():
    """Crear generadores de datos simples"""

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

def build_simple_model(num_classes):
    """Modelo CNN muy simple"""

    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    print("🚀 Iniciando entrenamiento simple...")

    # Crear generadores
    train_generator, val_generator = create_data_generators()
    num_classes = len(train_generator.class_indices)
    print(f"Clases encontradas: {num_classes}")
    print(f"Clases: {list(train_generator.class_indices.keys())}")

    # Construir modelo
    model = build_simple_model(num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('simple_waste_model.h5', save_best_only=True)
    ]

    # Entrenar
    print("\n🎯 Entrenando modelo...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Guardar modelo
    model.save('simple_waste_model_final.h5')
    print("✅ Modelo guardado!")

    # Evaluación simple
    loss, accuracy = model.evaluate(val_generator)
    print(f"📊 Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")
    print("🎉 Entrenamiento completado!")

if __name__ == "__main__":
    main()