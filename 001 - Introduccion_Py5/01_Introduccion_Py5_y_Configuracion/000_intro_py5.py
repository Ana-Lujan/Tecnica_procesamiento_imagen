import py5 # type: ignore
import os

def setup():
    py5.size(500, 500)
    py5.fill("#00D92F")
    py5.rect(150, 150, 200, 200)
    # Crear carpeta si no existe
    save_dir = "Imagenes/testing"
    os.makedirs(save_dir, exist_ok=True)
    py5.save(f"{save_dir}/000_intro_py5.png")

py5.run_sketch()
