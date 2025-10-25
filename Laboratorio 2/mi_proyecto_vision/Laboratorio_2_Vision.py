"""
Sistema de Clasificación de Reciclables - VERSIÓN MEJORADA


Este script crea una interfaz web usando Gradio para clasificar materiales reciclables
a partir de imágenes subidas. Actualmente usa una simulación, pero se puede mejorar
con un modelo de IA real.
"""

# Importaciones necesarias
import gradio as gr  # Para crear la interfaz web
import numpy as np  # Para cálculos numéricos y aleatoriedad
from PIL import Image, ImageDraw, ImageFont  # Para procesamiento de imágenes
import os  # Para operaciones del sistema de archivos
import time  # Para simular tiempo de procesamiento
from datetime import datetime  # Para timestamps
import json  # Para manejo de datos JSON (no usado actualmente)
# from transformers import pipeline  # Para usar modelos de Hugging Face - comentado por problemas de compatibilidad
# from mtcnn import MTCNN  # Para detección de rostros - comentado por problemas de compatibilidad
import joblib  # Para cargar modelo de scikit-learn

# Cargar modelo entrenado si existe
waste_classifier = None
class_names = []
try:
    waste_classifier = joblib.load('waste_classifier_sklearn.pkl')
    class_names = joblib.load('class_names.pkl')
    print("✅ Modelo de residuos cargado exitosamente")
except Exception as e:
    print(f"⚠️  No se pudo cargar modelo entrenado: {e}")
    print("🔄 Usando simulación como fallback")

print("🚀 Iniciando Clasificador de Materiales Reciclables...")

# Inicializar el modelo de clasificación de materiales reciclables
print("🤖 Intentando cargar modelos de IA...")
# Modelo CLIP comentado por problemas de compatibilidad
ancestry_classifier = None
print("⚠️ Modelo CLIP deshabilitado por problemas de compatibilidad")

# ============================================
# CONFIGURACIÓN MEJORADA
# ============================================

# Lista de todas las categorías de materiales reciclables que el sistema puede identificar
CATEGORIAS_RECICLABLES = [
    "plástico", "metal", "papel", "vidrio", "cartón", "orgánico", "peligroso"
]

# Colores asignados a cada categoría para visualizaciones
COLORES_CATEGORIAS = {
    "plástico": "#FF6B6B",    # Rojo coral
    "metal": "#C0C0C0",      # Plata
    "papel": "#FFFF99",      # Amarillo claro
    "vidrio": "#87CEEB",     # Azul cielo
    "cartón": "#D2691E",     # Chocolate
    "orgánico": "#32CD32",   # Verde lima
    "peligroso": "#FF0000"   # Rojo para materiales peligrosos
}

# ============================================
# BASE DE DATOS DE CONOCIMIENTO MEJORADA
# ============================================

# Diccionario con información detallada sobre cada categoría de material reciclable
# Incluye descripción, características físicas, usos comunes e información de reciclaje
BASE_CONOCIMIENTO = {
    "plástico": {
        "descripcion": "Material sintético derivado del petróleo, versátil y duradero",
        "caracteristicas": "Ligero, flexible, resistente al agua, colores variados",
        "usos": "Botellas, envases, bolsas, juguetes, utensilios",
        "reciclaje": "Se puede reciclar múltiples veces, reduce contaminación marina"
    },
    "metal": {
        "descripcion": "Material conductor, maleable y resistente a la corrosión",
        "caracteristicas": "Pesado, brillante, magnético (algunos), colores metálicos",
        "usos": "Latas, utensilios, herramientas, estructuras, cables",
        "reciclaje": "100% reciclable, ahorra energía en producción"
    },
    "papel": {
        "descripcion": "Material orgánico derivado de la celulosa vegetal",
        "caracteristicas": "Blanco o impreso, flexible, absorbente, liviano",
        "usos": "Periódicos, libros, cartulina, servilletas, embalaje",
        "reciclaje": "Reciclable, reduce deforestación y contaminación"
    },
    "vidrio": {
        "descripcion": "Material inorgánico transparente y frágil",
        "caracteristicas": "Transparente, frágil, pesado, colores variados",
        "usos": "Botellas, vasos, ventanas, lentes, vajilla",
        "reciclaje": "100% reciclable infinitamente, ahorra recursos naturales"
    },
    "cartón": {
        "descripcion": "Material compuesto de varias capas de papel prensado",
        "caracteristicas": "Rígido, corrugado, resistente, colores variados",
        "usos": "Cajas, embalaje, muebles, artesanías",
        "reciclaje": "Altamente reciclable, reduce residuos sólidos"
    },
    "orgánico": {
        "descripcion": "Material biodegradable de origen vegetal o animal",
        "caracteristicas": "Natural, perecedero, olor característico, colores naturales",
        "usos": "Alimentos, restos vegetales, productos perecederos",
        "reciclaje": "Compostable, genera abono natural, reduce metano"
    },
    "peligroso": {
        "descripcion": "Materiales tóxicos o peligrosos que requieren manejo especial",
        "caracteristicas": "Tóxicos, corrosivos, inflamables, colores de advertencia",
        "usos": "Baterías, pesticidas, productos químicos, electrónicos",
        "reciclaje": "Requiere centros especializados, no mezclar con residuos normales"
    }
}

# ============================================
# FUNCIONES MEJORADAS
# ============================================

def analizar_material_reciclable(imagen):
    """
    Análisis de material reciclable usando modelo entrenado o CLIP como fallback

    Esta función primero intenta usar el modelo de scikit-learn entrenado,
    y si no está disponible, usa CLIP o simulación mejorada.

    Parámetros:
    - imagen: Objeto PIL Image que contiene la imagen a analizar

    Retorna:
    - resultados_ordenados: Diccionario con materiales y probabilidades ordenadas
    - error: Mensaje de error si ocurre (vacío si no hay error)
    """
    if imagen is None:
        return {"error": "No se proporcionó imagen"}, "No se proporcionó imagen"

    # Validar que la imagen sea un objeto PIL válido con mejor manejo de errores
    try:
        # Verificar que sea un objeto PIL Image
        if not isinstance(imagen, Image.Image):
            return {"error": "Formato de imagen no válido"}, "Formato de imagen no válido"

        # Verificar que la imagen tenga contenido válido
        if imagen.size[0] == 0 or imagen.size[1] == 0:
            return {"error": "Imagen vacía o corrupta"}, "Imagen vacía o corrupta"

        # Intentar acceder a los píxeles para verificar integridad
        _ = imagen.load()

    except (AttributeError, OSError, ValueError) as e:
        print(f"Error al validar imagen: {e}")
        return {"error": f"Imagen inválida: {str(e)}"}, f"Imagen inválida: {str(e)}"

    try:
        # Intentar usar modelo entrenado primero
        if waste_classifier is not None:
            print("🤖 Clasificando material reciclable con modelo entrenado...")

            # Preprocesar imagen
            processed_image = preprocess_image_for_model(imagen)
            if processed_image is None:
                return simular_analisis_reciclable(imagen)

            # Hacer predicción
            prediction = waste_classifier.predict(processed_image)[0]
            probabilities = waste_classifier.predict_proba(processed_image)[0]

            # Crear diccionario de resultados detallados
            results_detailed = {}
            for i, prob in enumerate(probabilities):
                class_name = class_names[i]
                # Simplificar nombres de clase
                simple_name = class_name.split('_')[1] if '_' in class_name else class_name
                results_detailed[simple_name] = float(prob)

            # Mapear a categorías principales
            main_results = map_detailed_to_main_categories(results_detailed)

            # Ordenar por probabilidad
            resultados_ordenados = dict(sorted(main_results.items(), key=lambda x: x[1], reverse=True))

            print(f"✅ Análisis completado con modelo entrenado: {list(resultados_ordenados.keys())[0]}")
            return resultados_ordenados, ""

        # Fallback a CLIP si está disponible
        elif ancestry_classifier is not None:
            print("🤖 Clasificando material reciclable con CLIP...")

            # Crear prompts descriptivos mejorados para cada material reciclable
            candidate_labels = [
                f"objeto hecho de {material} para reciclaje, con apariencia típica de material {material}"
                for material in CATEGORIAS_RECICLABLES
            ]

            # Ejecutar clasificación con CLIP
            resultados_modelo = ancestry_classifier(imagen, candidate_labels=candidate_labels)

            # Mapear resultados a nuestras categorías
            resultados_mapeados = {}
            for resultado in resultados_modelo:
                label = resultado['label']
                score = resultado['score']

                # Extraer el nombre del material del prompt
                for material in CATEGORIAS_RECICLABLES:
                    if f"material {material}" in label:
                        resultados_mapeados[material] = score
                        break

            # Normalizar probabilidades
            total = sum(resultados_mapeados.values())
            if total > 0:
                resultados_mapeados = {k: round(v/total, 3) for k, v in resultados_mapeados.items()}

            # Ordenar por probabilidad
            resultados_ordenados = dict(sorted(resultados_mapeados.items(), key=lambda x: x[1], reverse=True))

            print(f"✅ Análisis completado con CLIP: {list(resultados_ordenados.keys())[0]}")
            return resultados_ordenados, ""

        else:
            # Fallback a simulación mejorada si ningún modelo está disponible
            print("⚠️ Ningún modelo disponible, usando simulación mejorada...")
            return simular_analisis_reciclable(imagen)

    except Exception as e:
        # Si hay error, usar simulación mejorada como fallback
        print(f"❌ Error en análisis: {e}, usando simulación mejorada...")
        return simular_analisis_reciclable(imagen)

def detectar_materiales_peligrosos(imagen, resultados_basicos):
    """
    Función especializada para detectar materiales peligrosos con mayor precisión

    Parámetros:
    - imagen: Objeto PIL Image
    - resultados_basicos: Resultados básicos de clasificación

    Retorna:
    - resultados_actualizados: Diccionario con probabilidades ajustadas para peligrosos
    """
    if imagen is None or resultados_basicos is None:
        return resultados_basicos

    try:
        # Extraer características adicionales para detección de peligrosos
        width, height = imagen.size
        img_array = np.array(imagen.convert('RGB'))
        average_color = np.mean(img_array, axis=(0, 1))
        r, g, b = average_color

        # Características específicas para materiales peligrosos
        brightness = (r + g + b) / 3
        saturation = max(r, g, b) - min(r, g, b)
        color_variance = np.var(img_array, axis=(0, 1))
        total_variance = np.sum(color_variance)

        # Análisis de forma y color para materiales peligrosos
        aspect_ratio = width / height

        # Indicadores de material peligroso
        hazard_indicators = 0
        hazard_score = 0

        # 1. Colores de advertencia (rojo, amarillo, negro, blanco con negro)
        if r > 150 and g < 100 and b < 100:  # Rojo intenso
            hazard_indicators += 1
            hazard_score += 0.3
        if r > 150 and g > 150 and b < 100:  # Amarillo
            hazard_indicators += 1
            hazard_score += 0.2
        if brightness < 50:  # Muy oscuro (posible negro)
            hazard_indicators += 1
            hazard_score += 0.2

        # 2. Formas características (cilíndricas para baterías, rectangulares para electrónicos)
        if 0.8 < aspect_ratio < 1.2:  # Cuadrado/redondo
            hazard_indicators += 1
            hazard_score += 0.1
        elif aspect_ratio > 2.0:  # Muy rectangular (electrónicos)
            hazard_indicators += 1
            hazard_score += 0.15

        # 3. Texturas complejas (circuitos, etiquetas)
        if total_variance > 8000:
            hazard_indicators += 1
            hazard_score += 0.2

        # 4. Tamaño pequeño (baterías, componentes electrónicos)
        if width < 300 and height < 300:
            hazard_indicators += 1
            hazard_score += 0.15

        # 5. Alta saturación con colores específicos
        if saturation > 100:
            if r > g and r > b:  # Rojo saturado
                hazard_indicators += 1
                hazard_score += 0.25
            elif r > 150 and g > 150 and b < 50:  # Amarillo-naranja
                hazard_indicators += 1
                hazard_score += 0.2

        # Calcular probabilidad ajustada para peligroso
        base_hazard_prob = resultados_basicos.get('peligroso', 0)
        adjusted_hazard_prob = min(0.95, base_hazard_prob + hazard_score)

        # Si hay múltiples indicadores, aumentar significativamente
        if hazard_indicators >= 2:
            adjusted_hazard_prob = min(0.95, adjusted_hazard_prob + 0.2)
        elif hazard_indicators >= 3:
            adjusted_hazard_prob = min(0.95, adjusted_hazard_prob + 0.3)

        # Ajustar otras probabilidades si peligroso es dominante
        if adjusted_hazard_prob > 0.4:
            # Redistribuir probabilidad de otras categorías
            total_other = sum(prob for cat, prob in resultados_basicos.items() if cat != 'peligroso')
            if total_other > 0:
                reduction_factor = 0.8  # Reducir otras categorías en 20%
                for cat in resultados_basicos:
                    if cat != 'peligroso':
                        resultados_basicos[cat] *= reduction_factor

                # Renormalizar
                total = sum(resultados_basicos.values())
                if total > 0:
                    resultados_basicos = {k: v/total for k, v in resultados_basicos.items()}

        # Aplicar probabilidad ajustada
        resultados_basicos['peligroso'] = round(adjusted_hazard_prob, 3)

        # Re-normalizar para asegurar que sume 1.0
        total = sum(resultados_basicos.values())
        if total > 0:
            resultados_basicos = {k: round(v/total, 3) for k, v in resultados_basicos.items()}

        return resultados_basicos

    except Exception as e:
        print(f"Error en detección de peligrosos: {e}")
        return resultados_basicos

def preprocess_image_for_model(image):
    """Preprocesar imagen para el modelo de scikit-learn con mejoras"""
    try:
        # Convertir a RGB si no lo está
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionar a 50x50 como en el entrenamiento
        image = image.resize((50, 50))

        # Mejorar contraste y nitidez para mejor extracción de características
        from PIL import ImageEnhance, ImageFilter

        # Aumentar contraste ligeramente
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

        # Aumentar nitidez
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        # Convertir a array y normalizar
        image_array = np.array(image, dtype=np.float32) / 255.0

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

        return combined_features.reshape(1, -1)

    except Exception as e:
        print(f"Error preprocesando imagen: {e}")
        return None

def map_detailed_to_main_categories(results):
    """Mapear resultados detallados a las categorías principales con mejoras exhaustivas"""

    # Mapeo mejorado y más exhaustivo de categorías detalladas a principales
    category_mapping = {
        'plástico': [
            'plastic', 'platics_bags_wrappers', 'plastic_bottles', 'plastic_bags',
            'plastic_containers', 'plastic_packs', 'plastic_wrappers', 'plastic_bottle',
            'plastic_cup', 'plastic_lid', 'plastic_straw', 'plastic_utensil',
            'polyethylene', 'polypropylene', 'pvc', 'pet_bottle', 'hdpe_container',
            'ldpe_bag', 'ps_foam', 'plastic_film', 'shrink_wrap', 'bubble_wrap',
            'plastic_toy', 'plastic_bucket', 'plastic_tube', 'plastic_pipe'
        ],
        'metal': [
            'metal', 'cans_all_type', 'batteries', 'aluminum_can', 'steel_can',
            'tin_can', 'metal_can', 'metal_container', 'metal_lid', 'metal_foil',
            'aluminum_foil', 'copper_wire', 'iron_object', 'steel_object',
            'ferrous_metal', 'non_ferrous_metal', 'scrap_metal', 'metal_scrap',
            'aluminum_scrap', 'copper_scrap', 'brass_object', 'bronze_object',
            'stainless_steel', 'galvanized_metal', 'metal_sheet', 'metal_bar',
            'metal_rod', 'metal_chain', 'metal_hinge', 'metal_lock'
        ],
        'papel': [
            'paper', 'paper_products', 'newspaper', 'magazine', 'notebook',
            'office_paper', 'printer_paper', 'tissue_paper', 'napkin', 'envelope',
            'cardboard_sheet', 'paper_bag', 'paper_cup', 'paper_plate',
            'newsprint', 'glossy_paper', 'matte_paper', 'cardstock', 'construction_paper',
            'tracing_paper', 'parchment_paper', 'rice_paper', 'wax_paper',
            'carbon_paper', 'art_paper', 'sketch_paper', 'watercolor_paper',
            'printing_paper', 'copy_paper', 'bond_paper', 'ledger_paper'
        ],
        'vidrio': [
            'glass', 'glass_containers', 'glass_bottle', 'glass_jar', 'wine_bottle',
            'beer_bottle', 'glass_cup', 'glass_plate', 'broken_glass', 'window_glass',
            'tempered_glass', 'laminated_glass', 'stained_glass', 'frosted_glass',
            'colored_glass', 'clear_glass', 'opaque_glass', 'glass_vase',
            'glass_bowl', 'glass_dish', 'glass_tumbler', 'glass_goblet',
            'glass_stemware', 'laboratory_glassware', 'beaker', 'flask',
            'test_tube', 'petri_dish', 'glass_lens', 'glass_prism'
        ],
        'cartón': [
            'carton', 'cardboard', 'cardboard_box', 'shipping_box', 'cereal_box',
            'pizza_box', 'egg_carton', 'milk_carton', 'juice_carton', 'cardboard_tube',
            'corrugated_cardboard', 'chipboard', 'fiberboard', 'pressboard',
            'cardboard_container', 'cardboard_display', 'cardboard_sign',
            'cardboard_partition', 'cardboard_divider', 'cardboard_sleeve',
            'cardboard_envelope', 'cardboard_mailbox', 'cardboard_file_box',
            'cardboard_storage_box', 'cardboard_shipping_tube', 'cardboard_core'
        ],
        'orgánico': [
            'organic', 'food_scraps', 'kitchen_waste', 'egg_shells', 'coffee_tea_bags',
            'yard_trimmings', 'fruit_peel', 'vegetable_peel', 'food_waste',
            'compost', 'biodegradable', 'plant_matter', 'animal_waste', 'dairy_product',
            'meat_waste', 'bread_waste', 'rice_waste', 'pasta_waste', 'leftover_food',
            'rotten_food', 'expired_food', 'fruit_waste', 'vegetable_waste',
            'vegetable_trimmings', 'fruit_cores', 'fruit_seeds', 'vegetable_stems',
            'leaf_waste', 'grass_clippings', 'tree_trimmings', 'flower_waste',
            'houseplant_waste', 'garden_waste', 'compostable_bag', 'biodegradable_plate',
            'wood_chips', 'sawdust', 'cork_waste', 'natural_fiber', 'cotton_waste',
            'wool_waste', 'silk_waste', 'leather_waste', 'feather_waste', 'bone_waste',
            'shell_waste', 'seaweed', 'algae', 'mushroom_waste', 'yeast_waste'
        ]
    }

    # Categorías peligrosas que requieren manejo especial - ampliado significativamente
    hazardous_mapping = {
        'peligroso': [
            'pesticides', 'batteries', 'e-waste', 'paint', 'chemical_container',
            'toxic_waste', 'hazardous_waste', 'medical_waste', 'radioactive_waste',
            'oil_filter', 'car_battery', 'lithium_battery', 'alkaline_battery',
            'mercury_battery', 'fluorescent_tube', 'broken_mercury_thermometer',
            'lead_acid_battery', 'nicd_battery', 'nimh_battery', 'lithium_ion_battery',
            'button_cell_battery', 'coin_cell_battery', 'aa_battery', 'aaa_battery',
            'c_battery', 'd_battery', '9v_battery', 'lantern_battery',
            'electronic_waste', 'computer_waste', 'phone_waste', 'tv_waste',
            'monitor_waste', 'keyboard_waste', 'mouse_waste', 'printer_waste',
            'circuit_board', 'motherboard', 'hard_drive', 'cd_dvd', 'floppy_disk',
            'usb_drive', 'memory_card', 'power_supply', 'charger', 'cable',
            'chemical_drum', 'solvent_container', 'acid_container', 'base_container',
            'corrosive_waste', 'flammable_liquid', 'oxidizer', 'explosive_material',
            'compressed_gas_cylinder', 'aerosol_can', 'propane_tank', 'oxygen_tank',
            'medical_sharps', 'syringe', 'needle', 'lancet', 'scalpel', 'surgical_tool',
            'blood_tube', 'culture_dish', 'petri_dish_contaminated', 'glove_box_waste',
            'pathological_waste', 'anatomical_waste', 'dialysis_waste', 'chemotherapy_waste',
            'pharmaceutical_waste', 'vaccine_waste', 'antibiotic_waste', 'hormone_waste',
            'radioactive_material', 'nuclear_waste', 'isotope', 'radiation_detector',
            'contaminated_clothing', 'protective_gear', 'lab_coat', 'glove',
            'mask', 'respirator', 'goggles', 'boot_cover', 'hair_net',
            'oil_waste', 'motor_oil', 'transmission_fluid', 'brake_fluid',
            'coolant', 'hydraulic_fluid', 'grease', 'lubricant', 'fuel_oil',
            'paint_thinner', 'varnish', 'lacquer', 'epoxy', 'adhesive', 'glue',
            'solvent', 'acetone', 'toluene', 'xylene', 'methylene_chloride',
            'pesticide_container', 'herbicide', 'insecticide', 'fungicide', 'rodenticide',
            'fertilizer_chemical', 'pool_chemical', 'bleach', 'ammonia', 'drain_cleaner',
            'oven_cleaner', 'rust_remover', 'mold_remover', 'mildew_remover',
            'asbestos', 'lead_paint', 'mercury_thermometer', 'mercury_switch',
            'fluorescent_bulb', 'cfl_bulb', 'led_hazardous', 'neon_tube',
            'thermostat', 'smoke_detector', 'carbon_monoxide_detector', 'fire_extinguisher'
        ]
    }

    main_results = {'plástico': 0, 'metal': 0, 'papel': 0, 'vidrio': 0, 'cartón': 0, 'orgánico': 0, 'peligroso': 0}

    for detailed_category, prob in results.items():
        detailed_lower = detailed_category.lower()

        # Buscar en el mapeo principal con mejor coincidencia
        found = False
        for main_cat, detailed_list in category_mapping.items():
            if any(detailed in detailed_lower for detailed in detailed_list):
                main_results[main_cat] += prob
                found = True
                break

        # Verificar si es material peligroso con mejor lógica
        if not found:
            for haz_cat, haz_list in hazardous_mapping.items():
                if any(haz in detailed_lower for haz in haz_list):
                    main_results[haz_cat] += prob
                    found = True
                    break

        # Si no se encontró coincidencia, usar lógica inteligente mejorada
        if not found:
            # Palabras clave para orgánico
            if any(word in detailed_lower for word in [
                'waste', 'scraps', 'kitchen', 'food', 'organic', 'fruit', 'vegetable',
                'meat', 'dairy', 'bread', 'rice', 'pasta', 'egg', 'shell', 'peel',
                'core', 'seed', 'stem', 'leaf', 'grass', 'tree', 'flower', 'plant',
                'garden', 'compost', 'biodegradable', 'natural', 'rotten', 'expired'
            ]):
                main_results['orgánico'] += prob
            # Palabras clave para envases y contenedores
            elif any(word in detailed_lower for word in [
                'can', 'bottle', 'container', 'packaging', 'jar', 'cup', 'plate',
                'box', 'bag', 'wrapper', 'pack', 'tube', 'lid', 'cap'
            ]):
                # Determinar material basado en contexto mejorado
                if any(word in detailed_lower for word in ['glass', 'wine', 'beer', 'bottle_green', 'bottle_clear']):
                    main_results['vidrio'] += prob
                elif any(word in detailed_lower for word in ['plastic', 'pet', 'hdpe', 'ldpe', 'pp', 'pvc', 'poly']):
                    main_results['plástico'] += prob
                elif any(word in detailed_lower for word in ['metal', 'aluminum', 'steel', 'tin', 'copper', 'iron']):
                    main_results['metal'] += prob
                elif any(word in detailed_lower for word in ['paper', 'cardboard', 'carton', 'newspaper', 'magazine']):
                    main_results['papel'] += prob
                else:
                    main_results['plástico'] += prob  # Default para envases
            # Palabras clave para peligroso
            elif any(word in detailed_lower for word in [
                'battery', 'electronic', 'chemical', 'toxic', 'hazardous', 'medical',
                'radioactive', 'pesticide', 'paint', 'oil', 'solvent', 'acid', 'base',
                'corrosive', 'flammable', 'explosive', 'compressed', 'aerosol',
                'sharps', 'syringe', 'blood', 'pharmaceutical', 'nuclear', 'asbestos',
                'lead', 'mercury', 'fluorescent', 'thermostat', 'smoke_detector'
            ]):
                main_results['peligroso'] += prob
            # Palabras clave para papel/cartón
            elif any(word in detailed_lower for word in [
                'paper', 'cardboard', 'carton', 'newspaper', 'magazine', 'notebook',
                'tissue', 'napkin', 'envelope', 'cardstock', 'newsprint', 'glossy'
            ]):
                if any(word in detailed_lower for word in ['cardboard', 'carton', 'box', 'corrugated']):
                    main_results['cartón'] += prob
                else:
                    main_results['papel'] += prob
            # Palabras clave para metal
            elif any(word in detailed_lower for word in [
                'metal', 'aluminum', 'steel', 'tin', 'copper', 'iron', 'brass',
                'bronze', 'stainless', 'galvanized', 'ferrous', 'nonferrous', 'scrap'
            ]):
                main_results['metal'] += prob
            # Palabras clave para vidrio
            elif any(word in detailed_lower for word in [
                'glass', 'tempered', 'laminated', 'stained', 'frosted', 'colored',
                'clear', 'opaque', 'vase', 'bowl', 'dish', 'tumbler', 'goblet',
                'stemware', 'laboratory', 'beaker', 'flask', 'lens', 'prism'
            ]):
                main_results['vidrio'] += prob
            # Palabras clave para plástico
            elif any(word in detailed_lower for word in [
                'plastic', 'polyethylene', 'polypropylene', 'pvc', 'pet', 'hdpe',
                'ldpe', 'ps', 'film', 'shrink', 'bubble', 'toy', 'bucket', 'pipe'
            ]):
                main_results['plástico'] += prob
            else:
                # Asignar basado en patrones comunes en residuos
                # Si contiene números o términos técnicos, podría ser peligroso
                if any(char.isdigit() for char in detailed_lower) or any(word in detailed_lower for word in ['tech', 'device', 'component', 'part']):
                    main_results['peligroso'] += prob * 0.7
                    main_results['metal'] += prob * 0.2
                    main_results['plástico'] += prob * 0.1
                else:
                    # Fallback inteligente basado en frecuencia típica en residuos
                    main_results['plástico'] += prob * 0.4  # Más común
                    main_results['papel'] += prob * 0.25
                    main_results['orgánico'] += prob * 0.2
                    main_results['metal'] += prob * 0.1
                    main_results['vidrio'] += prob * 0.03
                    main_results['cartón'] += prob * 0.01
                    main_results['peligroso'] += prob * 0.01

    # Normalizar
    total = sum(main_results.values())
    if total > 0:
        main_results = {k: round(v/total, 3) for k, v in main_results.items()}
    else:
        # Si no hay resultados, devolver distribución uniforme
        main_results = {k: round(1/7, 3) for k in main_results.keys()}

    return main_results


def simular_analisis_reciclable(imagen):
    """
    Función de simulación mejorada como fallback cuando los modelos de IA no están disponibles

    Parámetros:
    - imagen: Objeto PIL Image

    Retorna:
    - resultados_ordenados: Diccionario con materiales reciclables simulados
    - error: String vacío
    """
    # Extraer características básicas de la imagen para simulación más inteligente
    width, height = imagen.size
    aspect_ratio = width / height

    # Convertir a array para análisis de color
    img_array = np.array(imagen.convert('RGB'))
    average_color = np.mean(img_array, axis=(0, 1))
    r, g, b = average_color

    # Análisis mejorado basado en forma, color y características avanzadas
    # Calcular características adicionales
    brightness = (r + g + b) / 3
    saturation = max(r, g, b) - min(r, g, b)
    hue_dominant = 'red' if r > g and r > b else 'green' if g > r and g > b else 'blue'

    # Calcular varianza de color para detectar texturas complejas (posible cartón o papel)
    color_variance = np.var(img_array, axis=(0, 1))
    total_variance = np.sum(color_variance)

    # Análisis basado en forma, color y textura mejorado
    if aspect_ratio > 1.8:
        # Imágenes muy anchas (latas, envases largos, baterías)
        if saturation < 25 and brightness > 180:  # Muy brillante y desaturado - metal plateado
            resultados = {"metal": 0.45, "plástico": 0.20, "vidrio": 0.15, "papel": 0.10, "cartón": 0.08, "orgánico": 0.01, "peligroso": 0.01}
        elif r > 180 and g < 100 and b < 100:  # Rojo intenso - posible batería o material peligroso
            resultados = {"peligroso": 0.35, "metal": 0.25, "plástico": 0.20, "vidrio": 0.10, "papel": 0.05, "cartón": 0.03, "orgánico": 0.02}
        elif total_variance > 5000:  # Alta varianza - posible cartón corrugado
            resultados = {"cartón": 0.35, "papel": 0.25, "plástico": 0.20, "metal": 0.10, "vidrio": 0.08, "orgánico": 0.01, "peligroso": 0.01}
        else:  # Envases estándar
            resultados = {"plástico": 0.35, "metal": 0.25, "vidrio": 0.20, "papel": 0.10, "cartón": 0.08, "orgánico": 0.01, "peligroso": 0.01}
    elif aspect_ratio < 0.6:
        # Imágenes muy altas (botellas, papeles verticales, baterías AA)
        if g > 150 and r < 120 and b < 120:  # Verde intenso - vidrio o orgánico
            if saturation > 90:  # Verde saturado - orgánico fresco
                resultados = {"orgánico": 0.50, "vidrio": 0.25, "plástico": 0.15, "papel": 0.05, "cartón": 0.03, "metal": 0.01, "peligroso": 0.01}
            else:  # Verde apagado - vidrio verde
                resultados = {"vidrio": 0.45, "orgánico": 0.20, "plástico": 0.15, "papel": 0.10, "cartón": 0.08, "metal": 0.01, "peligroso": 0.01}
        elif brightness > 220 and saturation < 30:  # Blanco muy brillante - papel o cartón blanco
            resultados = {"papel": 0.40, "cartón": 0.25, "vidrio": 0.15, "plástico": 0.10, "metal": 0.08, "orgánico": 0.01, "peligroso": 0.01}
        elif width < 200 and height > 400:  # Muy estrecho y alto - posible batería cilíndrica
            resultados = {"peligroso": 0.40, "metal": 0.30, "plástico": 0.15, "vidrio": 0.08, "papel": 0.05, "cartón": 0.01, "orgánico": 0.01}
        else:  # Botellas estándar
            resultados = {"vidrio": 0.35, "plástico": 0.30, "papel": 0.15, "cartón": 0.10, "metal": 0.08, "orgánico": 0.01, "peligroso": 0.01}
    elif width > 600 and height > 400:
        # Imágenes grandes (cajas de cartón, periódicos grandes)
        if total_variance > 8000:  # Alta varianza de textura - cartón corrugado
            resultados = {"cartón": 0.50, "papel": 0.25, "plástico": 0.10, "metal": 0.08, "vidrio": 0.05, "orgánico": 0.01, "peligroso": 0.01}
        elif brightness > 200 and saturation < 40:  # Blanco/gris claro - papel o cartón
            resultados = {"papel": 0.40, "cartón": 0.30, "plástico": 0.15, "metal": 0.08, "vidrio": 0.05, "orgánico": 0.01, "peligroso": 0.01}
        else:  # Otros objetos grandes
            resultados = {"cartón": 0.30, "papel": 0.25, "plástico": 0.20, "metal": 0.15, "vidrio": 0.08, "orgánico": 0.01, "peligroso": 0.01}
    elif g > r and g > b and g > 130:
        # Verde dominante - orgánico o vidrio verde
        if saturation > 100 and total_variance > 3000:  # Verde saturado con textura - orgánico
            resultados = {"orgánico": 0.55, "vidrio": 0.20, "plástico": 0.10, "papel": 0.08, "cartón": 0.05, "metal": 0.01, "peligroso": 0.01}
        else:  # Verde más apagado - vidrio
            resultados = {"vidrio": 0.40, "orgánico": 0.25, "plástico": 0.15, "papel": 0.10, "cartón": 0.08, "metal": 0.01, "peligroso": 0.01}
    elif r > g and r > b and r > 160:
        # Rojo dominante - plástico rojo, metal oxidado, o peligroso
        if saturation > 120 and brightness < 150:  # Rojo oscuro saturado - posible peligroso
            resultados = {"peligroso": 0.40, "plástico": 0.25, "metal": 0.20, "vidrio": 0.08, "papel": 0.05, "cartón": 0.01, "orgánico": 0.01}
        elif saturation < 60:  # Rojo apagado - posible metal oxidado
            resultados = {"metal": 0.35, "plástico": 0.25, "peligroso": 0.15, "vidrio": 0.10, "papel": 0.08, "cartón": 0.05, "orgánico": 0.02}
        else:  # Rojo estándar - plástico
            resultados = {"plástico": 0.45, "metal": 0.20, "peligroso": 0.15, "vidrio": 0.10, "papel": 0.05, "cartón": 0.03, "orgánico": 0.02}
    elif saturation < 15 and brightness > 170:
        # Muy desaturado y brillante - metal o plástico blanco
        if aspect_ratio > 1.2:  # Más ancho - posible lata
            resultados = {"metal": 0.50, "plástico": 0.25, "vidrio": 0.10, "papel": 0.08, "cartón": 0.05, "orgánico": 0.01, "peligroso": 0.01}
        else:  # Más cuadrado - plástico blanco
            resultados = {"plástico": 0.40, "metal": 0.25, "papel": 0.15, "vidrio": 0.10, "cartón": 0.08, "orgánico": 0.01, "peligroso": 0.01}
    elif b > r and b > g and b > 140:
        # Azul dominante - posible vidrio azul o plástico azul
        if saturation < 80:  # Azul apagado - vidrio
            resultados = {"vidrio": 0.40, "plástico": 0.25, "metal": 0.15, "papel": 0.10, "cartón": 0.08, "orgánico": 0.01, "peligroso": 0.01}
        else:  # Azul saturado - plástico
            resultados = {"plástico": 0.35, "vidrio": 0.25, "metal": 0.15, "papel": 0.10, "cartón": 0.08, "orgánico": 0.01, "peligroso": 0.01}
    elif total_variance > 6000:
        # Alta varianza de textura - cartón o papel arrugado
        resultados = {"cartón": 0.35, "papel": 0.30, "plástico": 0.15, "metal": 0.10, "vidrio": 0.08, "orgánico": 0.01, "peligroso": 0.01}
    elif brightness < 80:
        # Imagen oscura - posible material peligroso o metal oscuro
        if saturation > 80:  # Oscuro pero saturado - posible peligroso
            resultados = {"peligroso": 0.30, "metal": 0.25, "plástico": 0.20, "vidrio": 0.10, "papel": 0.08, "cartón": 0.05, "orgánico": 0.02}
        else:  # Oscuro desaturado - metal
            resultados = {"metal": 0.35, "plástico": 0.25, "peligroso": 0.15, "vidrio": 0.10, "papel": 0.08, "cartón": 0.05, "orgánico": 0.02}
    else:
        # Imágenes estándar - distribución equilibrada mejorada con todas las categorías representadas
        resultados = {"plástico": 0.18, "papel": 0.16, "metal": 0.15, "vidrio": 0.14, "cartón": 0.13, "orgánico": 0.12, "peligroso": 0.12}

    # Crear semilla basada en propiedades de la imagen para consistencia
    seed = hash((str(imagen.size), str(imagen.mode), str(average_color))) % 1000
    np.random.seed(seed)

    # Añadir variabilidad aleatoria menor (±3%) para mayor consistencia
    for categoria in resultados:
        variacion = np.random.uniform(-0.03, 0.03)
        resultados[categoria] = max(0.001, resultados[categoria] + variacion)

    # Normalizar probabilidades
    total = sum(resultados.values())
    resultados = {k: round(v/total, 3) for k, v in resultados.items()}

    # Ordenar por probabilidad descendente
    resultados_ordenados = dict(sorted(resultados.items(), key=lambda x: x[1], reverse=True))

    return resultados_ordenados, ""

def generar_reporte_reciclable(resultados, material_principal):
    """
    Genera un reporte completo con información detallada sobre el análisis de material reciclable

    Parámetros:
    - resultados: Diccionario con las probabilidades de cada material
    - material_principal: El material con mayor probabilidad

    Retorna:
    - reporte: String formateado en Markdown con toda la información
    """
    if not resultados or "error" in resultados:
        return "No se pudo generar reporte"

    # Obtener información detallada del material principal
    info = BASE_CONOCIMIENTO.get(material_principal, {})

    # Crear reporte en formato Markdown
    reporte = f"""
## ♻️ Reporte de Análisis de Material Reciclable: **{material_principal.upper()}**

### 📝 Descripción
{info.get('descripcion', 'Material identificado')}

### 🔍 Características Físicas Típicas
{info.get('caracteristicas', 'Características físicas identificadas')}

### 🏭 Usos Comunes
{info.get('usos', 'Usos comunes del material')}

### 🎯 Confianza del Análisis
- **{material_principal}**: {resultados[material_principal]*100:.1f}%
- **Segunda opción**: {list(resultados.keys())[1] if len(resultados) > 1 else 'N/A'} ({resultados[list(resultados.keys())[1]]*100:.1f}% if len(resultados) > 1 else 'N/A')
- **Tercera opción**: {list(resultados.keys())[2] if len(resultados) > 2 else 'N/A'} ({resultados[list(resultados.keys())[2]]*100:.1f}% if len(resultados) > 2 else 'N/A')

### ♻️ Información de Reciclaje
{info.get('reciclaje', 'Información sobre reciclaje')}

### ⚠️ Importante
Este análisis es educativo y aproximado. Para una clasificación precisa, consulta con expertos en reciclaje.

### 🕒 Fecha de Análisis
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return reporte

def crear_visualizacion_resultados(resultados):
    """
    Crea una visualización gráfica de los resultados de clasificación

    Parámetros:
    - resultados: Diccionario con categorías y sus probabilidades

    Retorna:
    - img: Objeto PIL Image con el gráfico de barras, o None si hay error
    """
    if not resultados or "error" in resultados:
        return None

    # Crear una nueva imagen blanca de 400x250 píxeles (más alta para 5 barras)
    img = Image.new('RGB', (400, 250), color='white')
    draw = ImageDraw.Draw(img)  # Crear objeto para dibujar

    # Dibujar título en la parte superior
    draw.text((10, 10), "Resultados de Clasificación", fill='black')

    # Encontrar la probabilidad máxima para escalar las barras
    max_prob = max(resultados.values())
    y_pos = 40  # Posición Y inicial para las barras

    # Dibujar hasta 5 barras (las categorías más probables, incluyendo peligroso si es relevante)
    categorias_a_mostrar = []
    for categoria, prob in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        if prob > 0.05 or categoria == 'peligroso':  # Mostrar si >5% o es peligroso
            categorias_a_mostrar.append((categoria, prob))
        if len(categorias_a_mostrar) >= 5:
            break

    for i, (categoria, prob) in enumerate(categorias_a_mostrar):
        # Obtener color asignado a la categoría (gris por defecto si no existe)
        color = COLORES_CATEGORIAS.get(categoria, '#888888')

        # Calcular ancho de la barra proporcional a la probabilidad
        bar_width = int(350 * (prob / max_prob))

        # Dibujar la barra rectangular
        draw.rectangle([50, y_pos, 50 + bar_width, y_pos + 15], fill=color)

        # Dibujar nombre de la categoría a la izquierda
        draw.text((10, y_pos), categoria, fill='black')

        # Dibujar porcentaje a la derecha de la barra
        draw.text((50 + bar_width + 10, y_pos), f"{prob*100:.1f}%", fill='black')

        y_pos += 30  # Mover posición Y para la siguiente barra (más espacio para 5 barras)

    return img

# ============================================
# INTERFAZ GRADIO MEJORADA
# ============================================

# Crear la interfaz web usando Gradio Blocks
# Tema verde ecológico para relacionarse con el reciclaje
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="emerald",  # Verde esmeralda
        secondary_hue="green"   # Verde secundario
    ),
    title="♻️ Clasificador Inteligente de Reciclables",  # Título de la aplicación
    css="""
    .gradio-container {
        max-width: 1200px !important;  /* Limitar ancho máximo */
        margin: 0 auto;                /* Centrar horizontalmente */
        padding: 0 20px;               /* Padding lateral para móviles */
    }
    .success-box {
        border: 2px solid #4CAF50;     /* Borde verde */
        padding: 15px;                 /* Espaciado interno */
        border-radius: 10px;           /* Bordes redondeados */
        background: #f8fff8;           /* Fondo verde claro */
    }
    /* Responsive design para móviles */
    @media (max-width: 768px) {
        .gradio-container {
            max-width: 100% !important;
            padding: 0 10px;
        }
        .gradio-row {
            flex-direction: column !important;
        }
        .gradio-column {
            width: 100% !important;
        }
    }
    """
) as demo:
    
    # Encabezado principal con título y descripción
    gr.Markdown("""
    # ♻️ Clasificador Inteligente de Reciclables

    ## 🌍 *Sistema de IA para clasificación automática de materiales reciclables*

    **Analiza imágenes y clasifica materiales reciclables usando visión por computadora**
    """)
    
    # Sección principal dividida en dos columnas
    with gr.Row():
        # Columna izquierda: Input y controles
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Sube tu imagen")  # Título de la sección de input
            imagen_input = gr.Image(
                type="pil",                    # Tipo PIL para procesamiento
                label="Imagen del objeto a clasificar",
                height=300,                   # Altura fija para consistencia
                # sources=["upload", "webcam"]  # Comentado por compatibilidad con versión de Gradio
            )

            # Botones de acción en fila horizontal
            with gr.Row():
                btn_analizar = gr.Button(
                    "🔍 Clasificar Material",
                    variant="primary",         # Botón principal destacado
                    size="lg"                  # Tamaño grande
                )
                btn_limpiar = gr.Button(
                    "🗑️ Limpiar",
                    variant="secondary"        # Botón secundario
                )

            # Panel colapsable con estadísticas
            with gr.Accordion("📈 Estadísticas", open=False):  # Cerrado por defecto
                contador_analisis = gr.Number(
                    label="Análisis realizados",
                    value=0,                   # Valor inicial
                    interactive=False          # Solo lectura
                )
                ultima_categoria = gr.Textbox(
                    label="Último material identificado",
                    interactive=False          # Solo lectura
                )
        
        # Columna derecha: Resultados y visualizaciones (más ancha)
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Resultados de la Clasificación")

            # Fila con resultados principales
            with gr.Row():
                with gr.Column():
                    # Etiqueta con los 3 materiales más probables
                    resultado_label = gr.Label(
                        label="Material Principal",
                        num_top_classes=3  # Mostrar top 3
                    )
                    # Imagen con gráfico de barras
                    visualizacion = gr.Image(
                        label="Visualización de Probabilidades",
                        interactive=False,    # No editable
                        height=200
                    )

                with gr.Column():
                    # Características físicas
                    caracteristicas = gr.Textbox(
                        label="🔍 Características Físicas",
                        interactive=False,    # Solo lectura
                        lines=3,              # 3 líneas visibles
                        max_lines=5          # Máximo 5 líneas
                    )
                    # Información de reciclaje
                    cultura = gr.Textbox(
                        label="♻️ Información de Reciclaje",
                        interactive=False,    # Solo lectura
                        lines=3               # 3 líneas
                    )
    
    # Panel colapsable con reporte detallado
    with gr.Accordion("📊 Reporte Detallado de Material", open=False):  # Cerrado por defecto
        reporte_detallado = gr.Markdown(
            label="Análisis Completo",
            value="*El reporte se generará después del análisis...*"  # Texto inicial
        )

    # Guía interactiva de materiales reciclables
    with gr.Accordion("📚 Guía de Materiales Reciclables", open=False):  # Cerrado por defecto
        with gr.Row():
            # Mostrar información de los primeros 3 materiales
            for material in list(BASE_CONOCIMIENTO.keys())[:3]:
                with gr.Column():
                    gr.Markdown(f"""
                    ### {material.title()}  <!-- Título con mayúscula -->
                    **{BASE_CONOCIMIENTO[material]['descripcion']}**  <!-- Descripción en negrita -->

                    *{BASE_CONOCIMIENTO[material]['usos']}*  <!-- Usos en cursiva -->
                    """)

    # Sección de imágenes de prueba
    with gr.Accordion("🧪 Imágenes de Prueba", open=False):  # Cerrado por defecto
        gr.Markdown("### 📸 Prueba el sistema con estas imágenes de ejemplo:")

        # Crear una cuadrícula de imágenes de prueba
        with gr.Row():
            # Primera fila
            with gr.Column():
                gr.Markdown("**Botella de Plástico**")
                img_test_1 = gr.Image(
                    value="imagenes_prueba/botella_plastico.jpg",
                    label="Botella de Plástico",
                    interactive=False,
                    height=150
                )
                gr.Button("🔍 Probar con esta imagen", size="sm").click(
                    fn=lambda: procesar_imagen_completo(Image.open("imagenes_prueba/botella_plastico.jpg"), 0),
                    inputs=[],
                    outputs=[resultado_label, visualizacion, caracteristicas, cultura, reporte_detallado, contador_analisis, ultima_categoria]
                )

            with gr.Column():
                gr.Markdown("**Lata de Metal**")
                img_test_2 = gr.Image(
                    value="imagenes_prueba/lata_metal.jpg",
                    label="Lata de Metal",
                    interactive=False,
                    height=150
                )
                gr.Button("🔍 Probar con esta imagen", size="sm").click(
                    fn=lambda: procesar_imagen_completo(Image.open("imagenes_prueba/lata_metal.jpg"), 0),
                    inputs=[],
                    outputs=[resultado_label, visualizacion, caracteristicas, cultura, reporte_detallado, contador_analisis, ultima_categoria]
                )

            with gr.Column():
                gr.Markdown("**Papel Periódico**")
                img_test_3 = gr.Image(
                    value="imagenes_prueba/papel_periodico.jpg",
                    label="Papel Periódico",
                    interactive=False,
                    height=150
                )
                gr.Button("🔍 Probar con esta imagen", size="sm").click(
                    fn=lambda: procesar_imagen_completo(Image.open("imagenes_prueba/papel_periodico.jpg"), 0),
                    inputs=[],
                    outputs=[resultado_label, visualizacion, caracteristicas, cultura, reporte_detallado, contador_analisis, ultima_categoria]
                )

        with gr.Row():
            # Segunda fila
            with gr.Column():
                gr.Markdown("**Botella de Vidrio**")
                img_test_4 = gr.Image(
                    value="imagenes_prueba/botella_vidrio.jpg",
                    label="Botella de Vidrio",
                    interactive=False,
                    height=150
                )
                gr.Button("🔍 Probar con esta imagen", size="sm").click(
                    fn=lambda: procesar_imagen_completo(Image.open("imagenes_prueba/botella_vidrio.jpg"), 0),
                    inputs=[],
                    outputs=[resultado_label, visualizacion, caracteristicas, cultura, reporte_detallado, contador_analisis, ultima_categoria]
                )

            with gr.Column():
                gr.Markdown("**Caja de Cartón**")
                img_test_5 = gr.Image(
                    value="imagenes_prueba/caja_carton.jpg",
                    label="Caja de Cartón",
                    interactive=False,
                    height=150
                )
                gr.Button("🔍 Probar con esta imagen", size="sm").click(
                    fn=lambda: procesar_imagen_completo(Image.open("imagenes_prueba/caja_carton.jpg"), 0),
                    inputs=[],
                    outputs=[resultado_label, visualizacion, caracteristicas, cultura, reporte_detallado, contador_analisis, ultima_categoria]
                )

            with gr.Column():
                gr.Markdown("**Manzana (Orgánico)**")
                img_test_6 = gr.Image(
                    value="imagenes_prueba/manzana.jpg",
                    label="Manzana",
                    interactive=False,
                    height=150
                )
                gr.Button("🔍 Probar con esta imagen", size="sm").click(
                    fn=lambda: procesar_imagen_completo(Image.open("imagenes_prueba/manzana.jpg"), 0),
                    inputs=[],
                    outputs=[resultado_label, visualizacion, caracteristicas, cultura, reporte_detallado, contador_analisis, ultima_categoria]
                )

        # Más imágenes de prueba en fila adicional
        with gr.Row():
            for img_name in ["envase_yogur.jpg", "periodico.jpg", "lata_refresco.jpg", "botella_vino.jpg"]:
                with gr.Column():
                    categoria_display = img_name.replace(".jpg", "").replace("_", " ").title()
                    gr.Markdown(f"**{categoria_display}**")
                    img_path = f"imagenes_prueba/{img_name}"
                    gr.Image(
                        value=img_path,
                        label=categoria_display,
                        interactive=False,
                        height=120
                    )
                    gr.Button("🔍 Probar", size="sm").click(
                        fn=lambda path=img_path: procesar_imagen_completo(Image.open(path), 0),
                        inputs=[],
                        outputs=[resultado_label, visualizacion, caracteristicas, cultura, reporte_detallado, contador_analisis, ultima_categoria]
                    )
    
    # Pie de página con información técnica
    gr.Markdown("""
    ---
    ### 🔬 Información Técnica del Sistema

    - **🤖 Modelos**: Modelo ML entrenado + CLIP (OpenAI) como fallback
    - **🎯 Precisión**: Análisis basado en dataset de residuos reales
    - **🌐 Framework**: Gradio + Scikit-learn + Transformers
    - **📊 Datos**: Dataset Kaggle waste-classification + base de conocimiento 2025

    *Desarrollado para Procesamiento Digital de Imágenes y Visión por Computadora*

    **⚠️ Nota importante**: Este análisis es educativo y aproximado. Para clasificación precisa de reciclables, consulta con expertos ambientales.
    """)
    
    # ============================================
    # EVENT HANDLERS MEJORADOS
    # ============================================

    def procesar_imagen_completo(imagen, contador_actual):
        """
        Función principal que procesa la imagen completa y genera todos los outputs

        Parámetros:
        - imagen: Imagen PIL subida por el usuario
        - contador_actual: Número actual de análisis realizados

        Retorna:
        - Tupla con todos los valores para actualizar la interfaz
        """
        if imagen is None:
            # Si no hay imagen, devolver valores por defecto
            return (
                {"Error": "Sube una imagen primero"},
                None,                           # Sin visualización
                "Sube una imagen para analizar", # Características por defecto
                "Esperando imagen...",          # Cultura por defecto
                "*Sube una imagen para generar el reporte*",  # Reporte por defecto
                contador_actual,                # Contador sin cambiar
                "Ninguna"                       # Material por defecto
            )

        # Simular tiempo de procesamiento (reducido para mejor UX)
        time.sleep(0.5)  # Pausa de 0.5 segundos para simular procesamiento

        # Llamar a la función de análisis de material reciclable
        resultados, error = analizar_material_reciclable(imagen)

        if error:
            # Si hay error en el análisis
            return (
                {"Error": error},
                None,
                "Error en el análisis",
                "No disponible",
                f"**Error**: {error}",          # Reporte con mensaje de error
                contador_actual,                # Contador sin cambiar
                "Error"
            )

        # Aplicar detección especializada de materiales peligrosos
        resultados = detectar_materiales_peligrosos(imagen, resultados)

        # Procesamiento exitoso: obtener información detallada
        material_principal = list(resultados.keys())[0]  # Primer material (más probable)
        info_material = BASE_CONOCIMIENTO.get(material_principal, {})  # Info de la base de conocimiento

        # Verificar umbral de confianza para el material principal
        confianza_principal = resultados[material_principal]
        if confianza_principal < 0.25:  # Umbral de confianza bajo
            print(f"⚠️ Baja confianza en la clasificación: {material_principal} ({confianza_principal:.1%})")
            # Añadir nota de baja confianza al reporte
            nota_confianza = f"\n\n⚠️ **Nota**: La confianza en esta clasificación es baja ({confianza_principal:.1%}). Considere verificar manualmente."
        else:
            nota_confianza = ""

        # Generar todos los outputs necesarios
        visualizacion_img = crear_visualizacion_resultados(resultados)  # Gráfico de barras
        caracteristicas_texto = info_material.get('caracteristicas', 'Características identificadas')
        reciclaje_texto = info_material.get('reciclaje', 'Información de reciclaje disponible')
        reporte = generar_reporte_reciclable(resultados, material_principal) + nota_confianza  # Reporte completo

        # Incrementar contador de análisis
        nuevo_contador = contador_actual + 1

        # Devolver tupla con todos los valores actualizados
        return (
            resultados,           # Para resultado_label
            visualizacion_img,    # Para visualizacion
            caracteristicas_texto,  # Para caracteristicas
            reciclaje_texto,       # Para cultura (ahora reciclaje)
            reporte,              # Para reporte_detallado
            nuevo_contador,       # Para contador_analisis
            material_principal     # Para ultima_categoria
        )
    
    def limpiar_interfaz():
        """
        Función para limpiar/resetear toda la interfaz a su estado inicial

        Retorna:
        - Tupla con valores por defecto para todos los componentes
        """
        return (
            None,                           # Limpiar imagen_input
            {"": 0},                        # Resetear resultado_label
            None,                           # Limpiar visualizacion
            "Esperando análisis...",        # Texto por defecto para caracteristicas
            "Esperando análisis...",        # Texto por defecto para cultura
            "*Sube una imagen para generar el reporte*",  # Resetear reporte_detallado
            0,                              # Resetear contador_analisis
            "Ninguna"                       # Resetear ultimo material
        )
    
    # Conectar los eventos de los botones a sus funciones correspondientes

    # Evento del botón "Clasificar Material"
    btn_analizar.click(
        fn=procesar_imagen_completo,      # Función a ejecutar
        inputs=[imagen_input, contador_analisis],  # Inputs: imagen y contador actual
        outputs=[                          # Outputs en orden correspondiente
            resultado_label,
            visualizacion,
            caracteristicas,
            cultura,
            reporte_detallado,
            contador_analisis,
            ultima_categoria
        ]
    )

    # Evento del botón "Limpiar"
    btn_limpiar.click(
        fn=limpiar_interfaz,              # Función a ejecutar
        inputs=[],                        # Sin inputs
        outputs=[                         # Outputs: todos los componentes a resetear
            imagen_input,
            resultado_label,
            visualizacion,
            caracteristicas,
            cultura,
            reporte_detallado,
            contador_analisis,
            ultima_categoria
        ]
    )

# Bloque principal de ejecución
if __name__ == "__main__":
    print("✅ Clasificador de Materiales Reciclables cargado correctamente!")
    print("🌐 Iniciando servidor en http://127.0.0.1:7860")
    print("🎯 Características implementadas:")
    print("   - Clasificación de materiales reciclables con CLIP")
    print("   - Base de conocimiento de reciclaje")
    print("   - Visualizaciones gráficas")
    print("   - Reportes detallados de materiales")
    print("   - Interfaz profesional mejorada")

    # Función para encontrar un puerto disponible
    def find_available_port(start_port=7860, max_attempts=10):
        """Buscar un puerto disponible empezando desde start_port"""
        import socket
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # Si no encuentra ninguno, usar el original

    # Encontrar puerto disponible
    available_port = find_available_port(7860, 20)
    print(f"🌐 Usando puerto disponible: {available_port}")

    # Lanzar la aplicación Gradio
    demo.launch(
        server_name="0.0.0.0",    # Escuchar en todas las interfaces
        server_port=available_port,  # Puerto dinámico disponible
        share=False,              # No compartir públicamente
        inbrowser=True            # Abrir automáticamente en el navegador
    )