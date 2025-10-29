"""
Crear un modelo simple de clasificación de residuos usando scikit-learn
Como alternativa al modelo de TensorFlow que tiene problemas de memoria
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración
DATASET_PATH = './'
IMAGE_SIZE = (50, 50)  # Muy pequeño para reducir memoria
MAX_SAMPLES_PER_CLASS = 100  # Limitar muestras por clase

def preprocess_image_for_training(img):
    """Preprocesar imagen exactamente como en la inferencia para consistencia"""

    # Convertir a RGB si no lo está
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Redimensionar a 50x50 como en el entrenamiento
    img = img.resize(IMAGE_SIZE)

    # Mejorar contraste y nitidez para mejor extracción de características
    from PIL import ImageEnhance, ImageFilter

    # Aumentar contraste ligeramente
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    # Aumentar nitidez
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)

    # Convertir a array y normalizar
    image_array = np.array(img, dtype=np.float32) / 255.0

    # Extraer características adicionales: estadísticas de color
    features = []

    # Estadísticas por canal de color
    for channel in range(3):  # RGB
        channel_data = image_array[:, :, channel]
        features.extend([
            np.mean(channel_data),      # Media
            np.std(channel_data),       # Desviación estándar
            np.min(channel_data),       # Mínimo
            np.max(channel_data),       # Máximo
            np.median(channel_data),    # Mediana
            np.var(channel_data)        # Varianza
        ])

    # Estadísticas globales
    gray_image = np.mean(image_array, axis=2)
    features.extend([
        np.mean(gray_image),      # Media en escala de grises
        np.std(gray_image),       # Desviación en escala de grises
        np.var(gray_image),       # Varianza en escala de grises
    ])

    # Aplanar la imagen original
    flattened = image_array.flatten()

    # Combinar características
    combined_features = np.concatenate([flattened, features])

    return combined_features

def load_images_from_directory(base_path, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
    """Cargar imágenes y convertirlas a arrays con características avanzadas"""

    images = []
    labels = []
    class_names = []

    # Recorrer las carpetas principales
    for main_class in ['Hazardous', 'Recyclable', 'Organic', 'Non-Recyclable']:
        main_path = os.path.join(base_path, main_class)
        if not os.path.exists(main_path):
            continue

        print(f"Procesando {main_class}...")

        # Para este dataset, las imágenes están en main_class/main_class/subfolders
        actual_path = os.path.join(main_path, main_class)
        if not os.path.exists(actual_path):
            continue

        # Recorrer subcarpetas
        for subfolder in os.listdir(actual_path):
            subfolder_path = os.path.join(actual_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            class_name = f"{main_class}_{subfolder}"
            if class_name not in class_names:
                class_names.append(class_name)

            class_index = class_names.index(class_name)
            samples_loaded = 0

            # Cargar imágenes de esta clase
            for img_file in os.listdir(subfolder_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                if samples_loaded >= max_samples_per_class:
                    break

                try:
                    img_path = os.path.join(subfolder_path, img_file)
                    img = Image.open(img_path)

                    # Usar el mismo preprocesamiento que en inferencia
                    processed_features = preprocess_image_for_training(img)
                    images.append(processed_features)
                    labels.append(class_index)

                    samples_loaded += 1

                except Exception as e:
                    continue

    if len(images) == 0:
        return np.array([]), np.array([]), class_names

    return np.array(images), np.array(labels), class_names

def main():
    print("🚀 Creando modelo de clasificación de residuos con scikit-learn...")

    # Cargar datos
    print("\n📊 Cargando imágenes...")
    X, y, class_names = load_images_from_directory(DATASET_PATH)

    print(f"✅ Datos cargados: {len(X)} imágenes, {len(class_names)} clases")
    print(f"📏 Dimensiones de imagen: {IMAGE_SIZE}")
    if len(X) > 0:
        print(f"🔢 Características por imagen: {len(X[0])}")
    print(f"🏷️  Clases: {class_names}")

    if len(X) == 0:
        print("❌ No se encontraron imágenes. Verifica la estructura del dataset.")
        return

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📈 Datos de entrenamiento: {X_train.shape[0]}")
    print(f"🧪 Datos de prueba: {X_test.shape[0]}")

    # Crear y entrenar modelo
    print("\n🏗️  Creando modelo Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    print("🎯 Entrenando modelo...")
    model.fit(X_train, y_train)

    # Evaluar modelo
    print("\n🔍 Evaluando modelo...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"📊 Precisión en entrenamiento: {train_score:.4f}")
    print(f"📊 Precisión en prueba: {test_score:.4f}")
    # Predicciones detalladas
    y_pred = model.predict(X_test)

    print("\n📊 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Guardar modelo
    print("\n💾 Guardando modelo...")
    joblib.dump(model, 'waste_classifier_sklearn.pkl')
    joblib.dump(class_names, 'class_names.pkl')

    print("✅ Modelo guardado como 'waste_classifier_sklearn.pkl'")
    print("✅ Nombres de clases guardados como 'class_names.pkl'")

    print("\n🎉 ¡Modelo creado exitosamente!")
    print(f"📊 Precisión en prueba: {test_score:.4f}")

if __name__ == "__main__":
    main()