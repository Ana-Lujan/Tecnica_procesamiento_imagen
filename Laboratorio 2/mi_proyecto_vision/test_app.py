"""
Aplicación de Clasificación de Imágenes de Residuos
Desarrollada para Procesamiento Digital de Imágenes y Visión por Computadora

Autor: Ana Luján (Versión Corregida para CLIP)
Año: 2025
"""

import gradio as gr
from transformers import pipeline
from PIL import Image
# import numpy as np # Ya no es estrictamente necesario
# import os # Ya no es estrictamente necesario
# import random # Ya no es estrictamente necesario

# ============================================
# CONFIGURACIÓN DEL MODELO
# ============================================

print("Cargando modelo...")

# 🛑 SOLUCIÓN: Solo cargar CLIP para clasificación zero-shot
# CLIP es ideal porque NO NECESITA entrenamiento específico para tus categorías.
# Simplemente le pasamos las etiquetas de texto (zero-shot).
modelo = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)

# Opción 2: ViT para clasificación general (COMENTADO - NO USAR CON ZERO-SHOT)
# modelo = pipeline(
#     "image-classification",
#     model="google/vit-base-patch16-224"
# )

print("Modelo CLIP cargado correctamente")

# ============================================
# CONFIGURACIÓN DE CATEGORÍAS Y RECOMENDACIONES
# ============================================

# 🛑 SOLUCIÓN: Usaremos esta lista como 'candidate_labels' para CLIP
CATEGORIAS_RECICLABLES = [
    "plástico", 
    "vidrio", 
    "papel", 
    "cartón",
    "metal", 
    "orgánico", 
    "textil", 
    "electrónico", 
    "peligroso"
]

# Diccionario con recomendaciones específicas para cada categoría
RECOMENDACIONES = {
    "plástico": "♻️ CLASIFICADO: Plástico. Lava y seca los envases. Separa por tipos (PET, HDPE, etc.).",
    "vidrio": "♻️ CLASIFICADO: Vidrio. Enjuaga las botellas. Retira tapas metálicas. No incluyas cristales rotos.",
    "papel": "♻️ CLASIFICADO: Papel. Separa el papel limpio. Evita grasas, plastificado o muy sucio.",
    "cartón": "♻️ CLASIFICADO: Cartón. Aplana las cajas. Retira cintas, grapas y plásticos.",
    "metal": "♻️ CLASIFICADO: Metal. Enjuaga latas de comida y bebidas. Aplástalas para ahorrar espacio.",
    "orgánico": "🌱 CLASIFICADO: Orgánico. Ideal para compostaje casero o industrial. Separa completamente de otros residuos.",
    "textil": "👕 CLASIFICADO: Textil. Dona si está en buen estado. Si no, lleva a punto de reciclaje textil.",
    "electrónico": "💻 CLASIFICADO: Electrónico. Lleva a punto limpio especializado. No tires a la basura común.",
    "peligroso": "⚠️ CLASIFICADO: Peligroso. Maneja con cuidado. Lleva a punto limpio autorizado."
}

# ============================================
# FUNCIONES DE CLASIFICACIÓN (Modelo Real)
# ============================================

def clasificar_imagen(imagen):
    """
    Procesa una imagen usando el modelo CLIP (zero-shot) y retorna las predicciones.
    
    Args:
        imagen: Imagen PIL
    
    Returns:
        dict: Diccionario con categorías y probabilidades
    """
    if imagen is None:
        return {"Error": 1.0}
    
    try:
        # 🛑 SOLUCIÓN: Usar CATEGORIAS_RECICLABLES como las etiquetas candidatas
        resultados = modelo(imagen, candidate_labels=CATEGORIAS_RECICLABLES)
        
        # Formatear resultados
        return {
            resultado['label']: float(resultado['score'])
            for resultado in resultados
        }
    
    except Exception as e:
        print(f"Error en clasificación: {e}")
        # Retornar un error legible si falla
        return {"Error en el modelo": 1.0}

def obtener_recomendacion(categoria):
    """
    Obtiene la recomendación específica para una categoría de reciclable
    """
    return RECOMENDACIONES.get(
        categoria.lower(), 
        "Consulta las normas locales. Categoría no reconocida."
    )

def procesar_imagen(imagen):
    """
    Función principal que procesa la imagen, clasifica y genera resultados y recomendación.
    
    Parámetros:
    - imagen: Objeto PIL Image subido por el usuario
    
    Retorna:
    - tuple: (resultados_clasificacion, recomendacion_texto)
    """
    # 🛑 SOLUCIÓN: Llamar a la función que usa el modelo real
    resultados = clasificar_imagen(imagen)

    # Si hay error
    if "Error en el modelo" in resultados or "Error" in resultados:
        return resultados, "❌ Error al procesar la imagen. Verifica el formato o el estado del modelo."

    # Obtener la categoría con mayor probabilidad
    # El modelo de HF ya devuelve la lista ordenada por probabilidad descendente,
    # por lo que la primera clave es la más probable.
    categoria_principal = list(resultados.keys())[0]

    # Obtener recomendación específica para esa categoría
    recomendacion = obtener_recomendacion(categoria_principal)

    return resultados, recomendacion

# ============================================
# INTERFAZ DE USUARIO MEJORADA (Se mantiene)
# ============================================

# Crear la interfaz web usando Gradio Blocks
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="♻️ Clasificador de Reciclables - CLIP Zero-Shot"
) as demo:

    # Encabezado principal
    gr.Markdown("""
    # ♻️ Clasificador Zero-Shot de Residuos

    ## 🔬 Usando Modelo CLIP (Zero-Shot)

    **Instrucciones**: Sube una imagen para clasificarla en una de las 9 categorías de reciclables.
    """)

    # Sección principal dividida en columnas
    with gr.Row():
        # Columna izquierda: Input y controles
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Sube tu Imagen")

            # Input de imagen 
            imagen_input = gr.Image(
                type="pil",
                label="Imagen del residuo a clasificar",
                sources=["upload", "webcam"],
                height=300
            )

            # Botón principal de análisis
            btn_analizar = gr.Button(
                "🔍 Analizar Residuo",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            **💡 Consejos:** El modelo CLIP usa descripciones de texto para clasificar sin entrenamiento directo.
            """)

        # Columna derecha: Resultados
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Resultados del Análisis")

            # Resultados de clasificación
            # Mostrar todas las categorías
            resultado = gr.Label(
                label="Clasificación por Probabilidad",
                num_top_classes=len(CATEGORIAS_RECICLABLES), 
                color="green" 
            )

            # Recomendaciones de reciclaje
            recomendacion = gr.Textbox(
                label="💡 Recomendación de Reciclaje",
                interactive=False,
                lines=3,
                placeholder="Las recomendaciones aparecerán aquí después del análisis..."
            )

    # ============================================
    # EJEMPLOS (Necesitan rutas válidas para funcionar)
    # ============================================

    # Se recomienda mantener solo la estructura de ejemplos y asegurarte de tener las imágenes
    # en una carpeta "imagenes_prueba" o eliminar esta sección si no tienes las imágenes.
    # Por la limitación del entorno, esta sección se simplifica.

    # ============================================
    # INFORMACIÓN TÉCNICA Y AYUDA
    # ============================================

    with gr.Accordion("ℹ️ Información Técnica", open=False):
        gr.Markdown(f"""
        ### 🔬 Cómo Funciona
        
        Este clasificador usa el modelo **CLIP** (`openai/clip-vit-base-patch32`) en modo **Zero-Shot**. 
        Esto significa que el modelo compara la imagen con las siguientes etiquetas de texto:
        
        `{', '.join(CATEGORIAS_RECICLABLES)}`
        
        Y asigna una probabilidad a cada una sin haber sido entrenado específicamente en imágenes de residuos.

        ### 🎯 Categorías Soportadas ({len(CATEGORIAS_RECICLABLES)})
        
        | Categoría | Ejemplo de Recomendación |
        |-----------|-------------|
        | Plástico | {RECOMENDACIONES['plástico'].split('. ')[1]} |
        | Vidrio | {RECOMENDACIONES['vidrio'].split('. ')[1]} |
        | Cartón | {RECOMENDACIONES['cartón'].split('. ')[1]} |
        | Orgánico | {RECOMENDACIONES['orgánico'].split('. ')[1]} |
        | ... | ... |
        """)

    # ============================================
    # CONEXIÓN DE EVENTOS
    # ============================================

    # Evento del botón principal de análisis
    btn_analizar.click(
        fn=procesar_imagen,
        inputs=[imagen_input],
        outputs=[resultado, recomendacion]
    )

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("🎉 Aplicación de Clasificador de Residuos (CLIP) lista!")
    print("🌐 Iniciando servidor...")
    
    # Lanzar la aplicación Gradio
    demo.launch()