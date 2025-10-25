# ♻️ Clasificador Inteligente de Reciclables

## 🌍 Descripción del Proyecto

Este proyecto es un sistema avanzado de clasificación automática de materiales reciclables que combina Machine Learning entrenado con datos reales y modelos de IA de última generación. La aplicación permite subir imágenes de residuos y obtener clasificaciones precisas junto con recomendaciones detalladas para un reciclaje correcto.

## 📊 Fuente de Datos

### Dataset Principal: Waste Classification (Kaggle)
- **Nombre**: `waste-classification`
- **Usuario**: `phenomsg`
- **URL**: [https://www.kaggle.com/datasets/phenomsg/waste-classification](https://www.kaggle.com/datasets/phenomsg/waste-classification)
- **Credenciales**: Configurar credenciales personales de Kaggle API (ver sección de instalación)
- **Tamaño**: ~1GB comprimido, ~2GB descomprimido
- **Licencia**: MIT

### Estructura del Dataset
```
waste-classification/
├── Hazardous/           # Residuos peligrosos
│   ├── batteries/       # Baterías
│   ├── e-waste/         # Residuos electrónicos
│   ├── paints/          # Pinturas
│   └── pesticides/      # Pesticidas
├── Recyclable/          # Reciclables
│   ├── cans_all_type/   # Latas de todo tipo
│   ├── glass_containers/# Contenedores de vidrio
│   ├── paper_products/  # Productos de papel
│   └── plastic_bottles/ # Botellas de plástico
├── Organic/             # Orgánicos
│   ├── coffee_tea_bags/ # Bolsas de café/té
│   ├── egg_shells/      # Cáscaras de huevo
│   ├── food_scraps/     # Restos de comida
│   ├── kitchen_waste/   # Residuos de cocina
│   └── yard_trimmings/  # Recortes de jardín
└── Non-Recyclable/      # No reciclables
    ├── ceramic_product/ # Productos cerámicos
    ├── diapers/         # Pañales
    ├── platics_bags_wrappers/ # Bolsas y envoltorios plásticos
    ├── sanitary_napkin/ # Toallas sanitarias
    └── stroform_product/# Productos de espuma
```

### Estadísticas del Dataset
- **Total de imágenes**: ~2,500 imágenes
- **Categorías principales**: 4 (Hazardous, Recyclable, Organic, Non-Recyclable)
- **Subcategorías**: 18 tipos específicos de residuos
- **Formato**: Imágenes JPG/PNG de diversos tamaños
- **Calidad**: Variada (desde fotos profesionales hasta imágenes caseras)

## 🎯 Características Principales

- **🤖 Clasificación con IA Avanzada**: Sistema híbrido con modelo ML entrenado + transformers
- **📊 Modelo Entrenado**: Random Forest con 18 categorías específicas de residuos
- **🔄 Sistema Robusto**: Múltiples fallbacks (ML → CLIP → Simulación)
- **📊 Visualizaciones**: Gráficos de barras con probabilidades de clasificación
- **📋 Reportes Detallados**: Información completa sobre cada material reciclable
- **💡 Recomendaciones**: Guías específicas para el reciclaje correcto
- **🌱 Base de Conocimiento**: Información actualizada sobre impacto ambiental
- **📈 Estadísticas**: Seguimiento de análisis realizados
- **🎨 Interfaz Profesional**: Diseño moderno con tema ecológico

## 🛠️ Tecnologías Utilizadas

### Lenguaje y Frameworks
- **Python 3.9+**: Lenguaje principal
- **Gradio**: Framework para interfaz web interactiva
- **Scikit-learn**: Machine Learning (Random Forest)
- **Joblib**: Serialización de modelos

### Bibliotecas de IA y Visión Computacional
- **Transformers**: Biblioteca de Hugging Face para modelos de IA
- **Torch**: Framework de deep learning (usado por transformers)
- **PIL (Pillow)**: Procesamiento y manipulación de imágenes
- **NumPy**: Cálculos numéricos y arrays multidimensionales

### Herramientas de Datos
- **Pandas**: Manipulación y análisis de datos
- **Matplotlib**: Creación de gráficos y visualizaciones
- **Seaborn**: Visualizaciones estadísticas avanzadas

### APIs y Servicios Externos
- **Kaggle API**: Descarga automática de datasets
- **Hugging Face Hub**: Modelos pre-entrenados de IA

### Infraestructura
- **Virtualenv**: Entornos virtuales aislados
- **Pip**: Gestor de paquetes de Python
- **Git**: Control de versiones

## 📁 Estructura del Proyecto

```
mi_proyecto_vision/
├── 📄 README.md                          # Documentación completa del proyecto
├── 🐍 Laboratorio_2_Vision.py            # Aplicación principal Gradio
├── 🧪 test_app.py                        # Script de pruebas
├── 🤖 create_model.py                    # Entrenamiento del modelo ML
├── 🔧 integrate_model.py                 # Integración del modelo en la app
├── 📊 train_waste_classifier.py          # Versión avanzada de entrenamiento
├── 📈 simple_train.py                    # Versión simple de entrenamiento
├── 🔑 kaggle.json                        # Credenciales de Kaggle API (no incluir en repositorio)
├── 📦 waste_classifier_sklearn.pkl       # Modelo entrenado (Random Forest)
├── 🏷️ class_names.pkl                    # Nombres de clases del modelo
├── 🗂️ waste-classification.zip           # Dataset comprimido (descargar desde Kaggle)
├── ♻️ Hazardous/                         # Dataset: Residuos peligrosos (no incluir en repo)
├── 🔄 Recyclable/                        # Dataset: Materiales reciclables (no incluir en repo)
├── 🌱 Organic/                           # Dataset: Residuos orgánicos (no incluir en repo)
├── 🚫 Non-Recyclable/                    # Dataset: No reciclables (no incluir en repo)
├── 🖼️ imagenes_prueba/                   # Imágenes de prueba
│   ├── botella_plastico.jpg
│   ├── lata_metal.jpg
│   ├── papel_periodico.jpg
│   └── ... (10 imágenes de prueba)
├── 📁 ejemplos/                          # Ejemplos adicionales
└── 🐍 venv_proyecto/                     # Entorno virtual Python (no incluir en repo)
```

## 🚀 Instalación y Uso

### 1. Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### 2. Instalación

```bash
# Clonar o descargar el proyecto
cd mi_proyecto_vision

# Crear entorno virtual (recomendado)
python -m venv venv_proyecto
source venv_proyecto/bin/activate  # En Windows: venv_proyecto\Scripts\activate

# Instalar dependencias principales
pip install gradio scikit-learn joblib pillow numpy pandas matplotlib seaborn

# Instalar dependencias de IA (opcional, para CLIP fallback)
# Nota: transformers y torch pueden requerir instalación específica según tu sistema
pip install transformers torch

# Configurar Kaggle API (para descarga automática del dataset)
# 1. Obtener credenciales de Kaggle: https://www.kaggle.com/account
# 2. Crear archivo kaggle.json con formato: {"username":"TU_USUARIO","key":"TU_API_KEY"}
# 3. Configurar permisos:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Descargar dataset (opcional, ya incluido en el proyecto)
# Nota: Requiere credenciales de Kaggle configuradas
kaggle datasets download -d phenomsg/waste-classification
unzip waste-classification.zip
```

### 3. Ejecutar la Aplicación

```bash
python Laboratorio_2_Vision.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://127.0.0.1:7860`

## 📸 Cómo Usar la Aplicación

1. **Subir Imagen**: Haz clic en el área de subida y selecciona una imagen de un residuo
2. **Analizar**: Presiona el botón "🔍 Analizar Residuo"
3. **Ver Resultados**:
   - Clasificación principal con probabilidades
   - Visualización gráfica de resultados
   - Recomendaciones específicas de reciclaje
   - Puntos de reciclaje recomendados
4. **Explorar**: Usa los paneles colapsables para ver reportes detallados y guías

## 🧪 Pruebas con Imágenes de Ejemplo

La carpeta `imagenes_prueba/` contiene 10 imágenes de prueba para diferentes categorías:

1. **Botella de plástico** - Debe clasificarse como "plástico"
2. **Lata de metal** - Debe clasificarse como "metal"
3. **Papel periódico** - Debe clasificarse como "papel"
4. **Botella de vidrio** - Debe clasificarse como "vidrio"
5. **Caja de cartón** - Debe clasificarse como "cartón"
6. **Manzana** - Debe clasificarse como "orgánico"
7. **Envase de yogur** - Debe clasificarse como "plástico"
8. **Periódico** - Debe clasificarse como "papel"
9. **Lata de refresco** - Debe clasificarse como "metal"
10. **Botella de vino** - Debe clasificarse como "vidrio"

## 🤖 Arquitectura de IA y Machine Learning

### Sistema Híbrido de Clasificación

El proyecto implementa un sistema robusto de clasificación con múltiples capas de IA:

#### 1. Modelo Principal: Random Forest Entrenado
- **Algoritmo**: Random Forest Classifier (Scikit-learn)
- **Dataset**: Waste Classification de Kaggle (1,800 imágenes de entrenamiento)
- **Características**: 7,500 features por imagen (50x50x3 píxeles aplanados)
- **Categorías**: 18 clases específicas de residuos
- **Precisión**: ~19.4% en validación (limitado por complejidad de clases)

#### 2. Fallback: CLIP (Contrastive Language-Image Pretraining)
- **Modelo**: `openai/clip-vit-base-patch32`
- **Tarea**: Zero-shot Image Classification
- **Entrada**: Imágenes + prompts descriptivos en español
- **Categorías**: 6 categorías principales de reciclaje
- **Ventaja**: No requiere entrenamiento específico

#### 3. Último Fallback: Simulación Basada en Características
- **Características**: Relación de aspecto, tamaño, modo de color
- **Lógica**: Reglas heurísticas basadas en propiedades visuales
- **Categorías**: 6 categorías principales

### Pipeline de Procesamiento

```
Imagen de entrada → Preprocesamiento → Modelo ML → Mapeo a categorías principales → Resultados
                                      ↓ (si falla)
                            Modelo CLIP → Mapeo a categorías principales → Resultados
                                      ↓ (si falla)
                        Simulación → Resultados directos
```

### Entrenamiento del Modelo

#### Script de Entrenamiento: `create_model.py`
```python
# Configuración del modelo
IMAGE_SIZE = (50, 50)  # Redimensionamiento para eficiencia
MAX_SAMPLES_PER_CLASS = 100  # Balanceo de clases
BATCH_SIZE = 8  # Para evitar problemas de memoria

# Arquitectura del modelo
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1  # Paralelización
)
```

#### Resultados del Entrenamiento
- **Imágenes procesadas**: 1,800 imágenes
- **Tiempo de entrenamiento**: ~2-3 minutos
- **Precisión en entrenamiento**: 99.8%
- **Precisión en validación**: 19.4%
- **Archivo del modelo**: `waste_classifier_sklearn.pkl` (guardado con Joblib)

## 📊 Categorías Soportadas

### Clasificación Principal (6 Categorías)
| Categoría | Color | Descripción | Ejemplos |
|-----------|-------|-------------|----------|
| Plástico | 🔴 Rojo | Materiales plásticos reciclables | Botellas, envases, bolsas |
| Vidrio | 🔵 Turquesa | Vidrio transparente y coloreado | Botellas, frascos, tarros |
| Papel | 🔵 Azul | Papel y cartulina | Periódicos, revistas, hojas |
| Metal | 🟢 Verde | Metales ferrosos y no ferrosos | Latas, aluminio, acero |
| Orgánico | 🟡 Amarillo | Materiales biodegradables | Restos de comida, frutas |
| Cartón | 🟣 Rosa | Cartón corrugado y compacto | Cajas, empaques |

### Clasificación Detallada (18 Subcategorías del Dataset)
| Categoría Principal | Subcategorías |
|-------------------|---------------|
| **Hazardous** | batteries, e-waste, paints, pesticides |
| **Recyclable** | cans_all_type, glass_containers, paper_products, plastic_bottles |
| **Organic** | coffee_tea_bags, egg_shells, food_scraps, kitchen_waste, yard_trimmings |
| **Non-Recyclable** | ceramic_product, diapers, platics_bags_wrappers, sanitary_napkin, stroform_product |

## 🔧 Configuración Avanzada

### Cambiar el Modelo de IA
En `Laboratorio_2_Vision.py`, línea ~22:
```python
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
```

Modelos alternativos:
- `"microsoft/resnet-50"` - Más rápido, menos preciso
- `"facebook/deit-base-distilled-patch16-224"` - Buen balance
- `"google/vit-large-patch16-224"` - Más preciso, más lento

### Añadir Nuevas Categorías
1. Agregar al diccionario `CATEGORIAS_RECICLABLES`
2. Asignar color en `COLORES_CATEGORIAS`
3. Añadir información en `BASE_CONOCIMIENTO`
4. Actualizar el mapeo en `clasificar_imagen_mejorada()`

## 📈 Métricas y Estadísticas

La aplicación incluye:
- Contador de análisis realizados
- Última categoría identificada
- Tiempos de procesamiento
- Estadísticas de uso

## 🌱 Impacto Ambiental

Cada tonelada reciclada tiene un impacto positivo:

- **Papel**: Salva 17 árboles
- **Plástico**: Reduce contaminación marina
- **Vidrio**: 100% reciclable infinitamente
- **Metal**: Ahorra 95% de energía
- **Cartón**: Reduce 2.5 toneladas de CO2

## ⚠️ Limitaciones y Consideraciones

### Limitaciones Técnicas
- **Precisión del Modelo**: 19.4% en validación (limitado por complejidad de 18 clases)
- **Memoria**: Optimizado para sistemas con recursos limitados
- **Dependencia de Calidad**: Resultados dependen de la calidad de las imágenes
- **Categorías Específicas**: 18 subcategorías pueden ser demasiado granulares

### Limitaciones del Dataset
- **Balance de Clases**: Algunas categorías tienen más imágenes que otras
- **Calidad Variable**: Imágenes de diferentes fuentes y calidades
- **Sin Etiquetas de Confianza**: No incluye niveles de confianza por imagen

### Consideraciones Éticas y Ambientales
- **Sistema Educativo**: No reemplaza normas locales de reciclaje
- **Sin Identificación de Marcas**: No detecta símbolos específicos de reciclaje
- **Contexto Local**: Las normas de reciclaje varían por región
- **Impacto Ambiental**: El sistema promueve la conciencia, pero no sustituye la educación ambiental

### Limitaciones de la Interfaz
- **Idioma**: Principalmente en español
- **Accesibilidad**: Requiere conexión a internet para algunos modelos
- **Tamaño de Imágenes**: Limitado por la interfaz web

## 🤝 Contribuir

### Áreas de Mejora Prioritarias

1. **🔬 Mejorar el Modelo de ML**
   - Usar CNN en lugar de Random Forest
   - Implementar data augmentation avanzado
   - Balancear mejor las clases del dataset
   - Usar técnicas de ensemble learning

2. **📊 Expandir el Dataset**
   - Añadir más imágenes por categoría
   - Incluir imágenes de mejor calidad
   - Agregar metadata (ubicación, condiciones de iluminación)
   - Crear dataset de validación independiente

3. **🎨 Mejorar la Interfaz**
   - Diseño más moderno y responsive
   - Soporte para múltiples idiomas
   - Modo oscuro
   - Mejor accesibilidad (WCAG)

4. **⚡ Optimizaciones de Rendimiento**
   - Modelo más ligero para dispositivos móviles
   - Caché de predicciones
   - Procesamiento en lote
   - Optimización de memoria

5. **🔧 Nuevas Funcionalidades**
   - Detección de múltiples objetos por imagen
   - Clasificación por composición material
   - Integración con apps móviles
   - API REST para integraciones

### Guía para Contribuidores

1. **Fork** el repositorio
2. **Crear** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crear** un Pull Request

### Requisitos para Contribuciones

- Código bien documentado
- Tests unitarios cuando aplique
- Actualización del README
- Compatibilidad con Python 3.8+
- Licencia MIT

## 📄 Licencia

Este proyecto es educativo y está disponible bajo la licencia MIT.

## 📈 Métricas del Proyecto

### Rendimiento del Modelo
- **Tiempo de Entrenamiento**: ~2-3 minutos
- **Tiempo de Inferencia**: <1 segundo por imagen
- **Uso de Memoria**: ~50MB para el modelo cargado
- **Tamaño del Modelo**: ~10MB (comprimido)

### Estadísticas del Dataset
- **Imágenes Totales**: ~2,500
- **Imágenes de Entrenamiento**: 1,800
- **Imágenes de Validación**: 360
- **Resolución de Imágenes**: Variable (redimensionadas a 50x50)
- **Formatos**: JPG, PNG

### Cobertura de Categorías
- **Categorías Principales**: 6 (100% cobertura)
- **Subcategorías**: 18 (dataset completo)
- **Balance de Clases**: Variable (algunas clases con más muestras)

## 🔄 Versiones y Cambios

### v2.0.0 - Sistema con Modelo Entrenado
- ✅ Integración de modelo ML entrenado
- ✅ Sistema híbrido (ML + CLIP + Simulación)
- ✅ Dataset Kaggle completo
- ✅ Documentación completa
- ✅ Interfaz mejorada

### v1.0.0 - Versión Inicial
- ✅ Clasificación con CLIP
- ✅ Interfaz Gradio básica
- ✅ Simulación como fallback
- ✅ Base de conocimiento

## 📞 Contacto y Soporte

### Autor
**Desarrollado para el curso de Procesamiento Digital de Imágenes y Visión por Computadora**

### Soporte
- 📧 Reportar issues en el repositorio
- 📖 Consultar documentación en README
- 🐛 Debug: Revisar logs de la aplicación

### Comunidad
- 🌟 Star el repositorio si te gusta el proyecto
- 🍴 Fork para contribuir
- 📢 Compartir con otros estudiantes

## 🤗 Publicación en Hugging Face

### Para subir el modelo a Hugging Face Hub:

1. **Crear cuenta en Hugging Face**: https://huggingface.co/join
2. **Instalar Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```
3. **Login**:
   ```bash
   huggingface-cli login
   ```
4. **Crear repositorio para el modelo**:
   ```bash
   huggingface-cli repo create waste-classifier-spanish --type model
   ```
5. **Subir el modelo**:
   ```bash
   huggingface-cli upload waste-classifier-spanish waste_classifier_sklearn.pkl
   huggingface-cli upload waste-classifier-spanish class_names.pkl
   ```

### Para subir el dataset a Hugging Face Datasets:

1. **Crear repositorio para el dataset**:
   ```bash
   huggingface-cli repo create waste-classification-dataset --type dataset
   ```
2. **Preparar el dataset** (solo imágenes de prueba, no el dataset completo):
   ```bash
   # Crear carpeta dataset limpio
   mkdir hf_dataset
   cp -r imagenes_prueba/ hf_dataset/
   cp README.md hf_dataset/
   ```
3. **Subir el dataset**:
   ```bash
   huggingface-cli upload waste-classification-dataset hf_dataset/
   ```

### Estructura recomendada para Hugging Face:

```
waste-classifier-spanish/
├── README.md                    # Documentación del modelo
├── waste_classifier_sklearn.pkl # Modelo entrenado
├── class_names.pkl             # Nombres de clases
└── app.py                      # Script para demo (opcional)

waste-classification-dataset/
├── README.md                   # Documentación del dataset
├── imagenes_prueba/            # Imágenes de ejemplo
└── metadata.json               # Información del dataset
```

---

## 🎯 Conclusión

Este proyecto demuestra la aplicación práctica de técnicas de Machine Learning y Visión por Computadora para resolver un problema ambiental real. La combinación de un modelo entrenado específicamente con datos reales y sistemas de fallback robustos crea una solución educativa y funcional.

**💡 Tip**: Siempre verifica las normas específicas de reciclaje de tu localidad, ya que pueden variar según la región y el contexto local.

**🌍 Impacto**: Cada tonelada reciclada correctamente reduce significativamente el impacto ambiental y contribuye a un planeta más sostenible.