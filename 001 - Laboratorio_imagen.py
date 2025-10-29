
import py5

def setup():
    py5.size(200, 200)
    py5.background(200)
    py5.ellipse(100, 100, 100, 100)
    py5.save_frame("test.jpg")

py5.run_sketch()


