"""
Entrenamiento de modelo CNN para clasificación de residuos
Usando el dataset de Kaggle: waste-classification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
DATASET_PATH = './'
IMAGE_SIZE = (150, 150)  # Reducido para evitar problemas de memoria
BATCH_SIZE = 16  # Reducido para evitar problemas de memoria
EPOCHS = 20  # Reducido para pruebas rápidas

# Categorías principales del dataset
CATEGORIES = {
    'Hazardous': ['batteries', 'e-waste', 'paints', 'pesticides'],
    'Recyclable': ['cans_all_type', 'glass_containers', 'paper_products', 'plastic_bottles'],
    'Organic': ['coffee_tea_bags', 'egg_shells', 'food_scraps', 'kitchen_waste', 'yard_trimmings'],
    'Non-Recyclable': ['ceramic_product', 'diapers', 'platics_bags_wrappers', 'sanitary_napkin', 'stroform_product']
}

def create_data_generators():
    """Crear generadores de datos con aumentación"""

    # Generador de entrenamiento con aumentación
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Generador de validación (sin aumentación)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Generador de prueba
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Crear generadores
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

def build_model(num_classes):
    """Construir modelo CNN simplificado"""

    model = Sequential([
        # Primera capa convolucional
        Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Segunda capa convolucional
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Tercera capa convolucional
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Aplanamiento
        Flatten(),

        # Capas densas
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    return model

def plot_training_history(history):
    """Graficar el historial de entrenamiento"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Precisión
    ax1.plot(history.history['accuracy'], label='Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Validación')
    ax1.set_title('Precisión del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()

    # Pérdida
    ax2.plot(history.history['loss'], label='Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()

    # Precisión por clase (si está disponible)
    if 'categorical_accuracy' in history.history:
        ax3.plot(history.history['categorical_accuracy'], label='Entrenamiento')
        ax3.plot(history.history['val_categorical_accuracy'], label='Validación')
        ax3.set_title('Precisión Categórica')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Precisión')
        ax3.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Iniciando entrenamiento del clasificador de residuos...")
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"🖼️  Image size: {IMAGE_SIZE}")
    print(f"📦 Batch size: {BATCH_SIZE}")

    # Crear generadores de datos
    print("\n📊 Creando generadores de datos...")
    try:
        train_generator, val_generator = create_data_generators()
        num_classes = len(train_generator.class_indices)
        print(f"✅ Generadores creados exitosamente")
        print(f"📊 Número de clases: {num_classes}")
        print(f"🏷️  Clases: {list(train_generator.class_indices.keys())}")
        print(f"📈 Datos de entrenamiento: {train_generator.samples}")
        print(f"🧪 Datos de validación: {val_generator.samples}")
    except Exception as e:
        print(f"❌ Error al crear generadores: {e}")
        return

    # Construir modelo
    print("\n🏗️  Construyendo modelo CNN...")
    model = build_model(num_classes)
    model.summary()

    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_waste_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Entrenar modelo
    print("\n🎯 Iniciando entrenamiento...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Guardar modelo final
    model.save('waste_classifier_final.h5')
    print("💾 Modelo guardado como 'waste_classifier_final.h5'")

    # Graficar historial de entrenamiento
    print("\n📈 Generando gráficos de entrenamiento...")
    plot_training_history(history)

    # Evaluación final
    print("\n🔍 Evaluando modelo...")
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"📊 Pérdida de validación: {val_loss:.4f}")
    print(f"📊 Precisión de validación: {val_accuracy:.4f}")
    # Predicciones para métricas detalladas
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    # Reporte de clasificación
    class_names = list(val_generator.class_indices.keys())
    print("\n📊 Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✅ Entrenamiento completado exitosamente!")
    print("📁 Archivos guardados:")
    print("   - best_waste_classifier.h5 (mejor modelo)")
    print("   - waste_classifier_final.h5 (modelo final)")
    print("   - training_history.png (gráfico de entrenamiento)")
    print("   - confusion_matrix.png (matriz de confusión)")

if __name__ == "__main__":
    main()