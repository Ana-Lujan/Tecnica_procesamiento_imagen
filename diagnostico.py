#!/usr/bin/env python3
"""
Script de diagnóstico para identificar problemas en el clasificador de residuos
"""

import os
import sys
import traceback

def check_dependencies():
    """Verificar dependencias críticas"""
    print("🔍 Verificando dependencias...")

    dependencies = [
        ('gradio', 'Interfaz web'),
        ('PIL', 'Procesamiento de imágenes'),
        ('numpy', 'Cálculos numéricos'),
        ('joblib', 'Carga de modelos'),
    ]

    for module, description in dependencies:
        try:
            if module == 'PIL':
                import PIL
            else:
                __import__(module)
            print(f"✅ {description}: OK")
        except ImportError as e:
            print(f"❌ {description}: FALTA - {e}")
        except Exception as e:
            print(f"⚠️  {description}: ERROR - {e}")

def check_models():
    """Verificar modelos disponibles"""
    print("\n🤖 Verificando modelos...")

    models = [
        ('waste_classifier_sklearn.pkl', 'Modelo ML entrenado'),
        ('class_names.pkl', 'Nombres de clases'),
    ]

    for model_file, description in models:
        if os.path.exists(model_file):
            try:
                import joblib
                data = joblib.load(model_file)
                if model_file == 'waste_classifier_sklearn.pkl':
                    print(f"✅ {description}: OK ({type(data).__name__})")
                else:
                    print(f"✅ {description}: OK ({len(data)} clases)")
            except Exception as e:
                print(f"❌ {description}: ERROR al cargar - {e}")
        else:
            print(f"❌ {description}: ARCHIVO NO ENCONTRADO")

def check_images():
    """Verificar imágenes de prueba"""
    print("\n🖼️  Verificando imágenes de prueba...")

    test_dir = 'imagenes_prueba'
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✅ Carpeta {test_dir}: {len(images)} imágenes encontradas")

        # Probar cargar una imagen
        if images:
            try:
                from PIL import Image
                img_path = os.path.join(test_dir, images[0])
                img = Image.open(img_path)
                print(f"✅ Imagen de prueba cargada: {img.size} píxeles")
            except Exception as e:
                print(f"❌ Error al cargar imagen de prueba: {e}")
    else:
        print(f"❌ Carpeta {test_dir}: NO ENCONTRADA")

def check_transformers():
    """Verificar transformers (opcional)"""
    print("\n🔄 Verificando transformers (opcional)...")

    try:
        from transformers import pipeline
        print("✅ Transformers: OK")

        # Intentar crear pipeline CLIP
        try:
            model = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
            print("✅ Modelo CLIP: OK")
        except Exception as e:
            print(f"⚠️  Modelo CLIP: ERROR - {e}")

    except ImportError:
        print("⚠️  Transformers: NO INSTALADO (usará solo simulación)")
    except Exception as e:
        print(f"❌ Transformers: ERROR - {e}")

def test_basic_functionality():
    """Probar funcionalidad básica"""
    print("\n🧪 Probando funcionalidad básica...")

    try:
        from PIL import Image, ImageDraw
        import numpy as np

        # Crear imagen de prueba
        img = Image.new('RGB', (100, 100), color='red')
        print("✅ Creación de imagen PIL: OK")

        # Convertir a array
        arr = np.array(img)
        print(f"✅ Conversión a NumPy array: OK (shape: {arr.shape})")

        # Simular procesamiento
        processed = arr.flatten()
        print(f"✅ Aplanamiento de array: OK (shape: {processed.shape})")

    except Exception as e:
        print(f"❌ Error en funcionalidad básica: {e}")
        traceback.print_exc()

def main():
    print("🚀 Iniciando diagnóstico del Clasificador de Residuos")
    print("=" * 50)

    check_dependencies()
    check_models()
    check_images()
    check_transformers()
    test_basic_functionality()

    print("\n" + "=" * 50)
    print("✅ Diagnóstico completado")

if __name__ == "__main__":
    main()