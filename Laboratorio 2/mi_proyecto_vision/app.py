#!/usr/bin/env python3
"""
Aplicación principal para Hugging Face Spaces
Clasificador de residuos usando CLIP

Esta aplicación permite clasificar imágenes de residuos usando IA avanzada.
Sube una foto de un residuo y obtén recomendaciones de reciclaje personalizadas.

Autor: Ana Luján
Proyecto: Clasificador de Residuos - Visión por Computadora
"""

# Importar la aplicación desde test_app.py
from test_app import demo

if __name__ == "__main__":
    # Lanzar la aplicación optimizada para HF Spaces
    # Configuración para despliegue en la nube
    demo.launch(
        server_name="0.0.0.0",  # Acepta conexiones desde cualquier IP
        server_port=None,       # Puerto automático para evitar conflictos
        show_error=True,        # Muestra errores para debugging
        quiet=False             # Logging completo
    )