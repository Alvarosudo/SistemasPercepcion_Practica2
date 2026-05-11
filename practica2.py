import open3d as o3d
import numpy as np
import copy
import os
import time # Añadir en los imports


def preprocess_point_cloud(pcd, voxel_size,vecindad_normal,vecindad_descriptores, max_norm, max_vecin):
    """
    Realiza el filtrado (subsampling), cálculo de normales y cálculo de descriptores.
    """
    #(2 FIltrar para reducir tamano) 1. Filtrar la nube para reducir su tamaño (Voxel Downsampling)
    print(f"    -> Aplicando Voxel Downsampling con tamaño de voxel: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    #(3 Extraer puntos caracteristicos) 2. Extraer características (Cálculo de normales)
    print(f"    -> Calculando normales con radio de búsqueda: {vecindad_normal}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=vecindad_normal, max_nn=max_norm))

    #(4 Calcular los descriptores para los puntos característicos) 3. Calcular descriptores (FPFH - Fast Point Feature Histograms)
    print(f"    -> Calculando descriptores FPFH con radio: {vecindad_descriptores}")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=vecindad_descriptores, max_nn=max_vecin))
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size,umbral):
    """
    Alineación global inicial (Coarse alignment) usando RANSAC.
    """
    print(f"    -> Ejecutando RANSAC (Alineamiento global). Umbral de distancia: {umbral}")
    
    # RANSAC para alinear basándose en el emparejamiento de las características FPFH
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        umbral,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            # Comprobaciones para asegurar que los emparejamientos son lógicos geométricamente
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(umbral)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result

def refine_registration(source_down, target_down, initial_transform, voxel_size,umbral):
    """
    Refinamiento local (Fine alignment) usando ICP (Iterative Closest Point).
    """
    print(f"    -> Ejecutando ICP (Refinamiento). Umbral de distancia: {umbral}")
    
    # ICP (Point-to-Plane suele converger mejor y más rápido que Point-to-Point)
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, umbral, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return result

def main():
    cloud_seleccion = 1
    
    # Rutas de los archivos (ajusta las rutas según dónde tengas los ficheros descomprimidos)
    
    if cloud_seleccion == 1:
        SCENE_PATH = "clouds/scenes/snap_0point.pcd"
        OBJECTS_DIR = "clouds/objects/"
        object_files = ["s0_mug_corr.pcd", "s0_plc_corr.pcd", "s0_plant_corr.pcd", "s0_piggybank_corr.pcd"]
    elif cloud_seleccion == 2:
        SCENE_PATH = "clouds_new/clouds/scenes/snap_0point.pcd"
        OBJECTS_DIR = "clouds_new/clouds/objects/"
        object_files = ["s0_mug_corr.pcd", "s0_plc_corr.pcd", "s0_plant_corr.pcd", "s0_piggybank_corr.pcd"]
    elif cloud_seleccion == 3:
        SCENE_PATH = "more_clouds/clutter_scene/pcd_26.pcd"
        OBJECTS_DIR = "more_clouds/objects/"
        object_files = ["pcd_9.pcd", "pcd_21.pcd", "pcd_33.pcd"]
    elif cloud_seleccion == 4:
        SCENE_PATH = "more_clouds/pepper_scene/pcd_21.pcd"
        OBJECTS_DIR = "more_clouds/objects/"
        object_files = ["pcd_9.pcd", "pcd_21.pcd", "pcd_33.pcd"]

    
    
    # Parámetro clave: tamaño del voxel. Si los puntos están en metros, 0.005 son 5 milímetros.
    # ¡Prueba a cambiar este valor para tu memoria!
    #PARAMETROS INVOLUCRADOS EN EL TIMELINE
    VOXEL_SIZE = 0.005 #Tamano del voxel

    VECINDAD_NORMAL = VOXEL_SIZE * 2 #Vecindad para calcular las normales
    VECIN_NORM_MAX = 30 #Maximo de vecinos de la normal

    VECINDAD_DESCRIPTORES = VOXEL_SIZE * 5 #Vecindad para calcular los descriptores
    VECIN_DESC_MAX = 100 #Maximo de vecinos de los descriptores

    SAMPLES_RANSAC = 3 #Numero de samples para RANSAC
    UMBRAL_RANSAC = VOXEL_SIZE * 1.5 #Umbral de aceptacion para ransac
    UMBRAL_ICP = VOXEL_SIZE * 0.4 #Umbral de aceptacion para icp


    print("=== 1. CARGANDO Y PREPROCESANDO LA ESCENA ===")
    scene = o3d.io.read_point_cloud(SCENE_PATH)
    
    # Eliminar NaNs e infinitos de la escena provenientes del escáner
    scene = scene.remove_non_finite_points()

    #(1 Eliminar todos los planos dominantes de la escena)
    print("-> Eliminando el plano dominante de la escena (RANSAC plano)")
    # distance_threshold: tolerancia de distancia al plano
    plane_model, inliers = scene.segment_plane(distance_threshold=0.015,
                                               ransac_n=SAMPLES_RANSAC,
                                               num_iterations=1000)
    
    # Nos quedamos con la escena SIN el plano (invert=True)
    scene_no_plane = scene.select_by_index(inliers, invert=True)
    
    # Preprocesamos la escena (downsample, normales, FPFH)
    scene_down, scene_fpfh = preprocess_point_cloud(scene_no_plane, VOXEL_SIZE,VECINDAD_NORMAL,VECINDAD_DESCRIPTORES,VECIN_NORM_MAX,VECIN_DESC_MAX)

    # Nube final que usaremos para visualizar (una copia para no modificar la original)
    final_visualization = copy.deepcopy(scene)
    #final_visualization.paint_uniform_color([0.8, 0.8, 0.8]) # Escena en gris claro
    
    # Colores llamativos para cada objeto encontrado
    colors = [
        [1, 0, 0], # Rojo (Taza)
        [0, 1, 0], # Verde (PLC)
        [0, 0, 1], # Azul (Planta)
        [1, 1, 0]  # Amarillo (Hucha)
    ]

    print("\n=== 2. PROCESANDO OBJETOS Y BUSCÁNDOLOS EN LA ESCENA ===")
    
    for i, obj_filename in enumerate(object_files):
        print(f"\n--- Procesando: {obj_filename} ---")
        
        # 1. Cargar el objeto
        obj_path = os.path.join(OBJECTS_DIR, obj_filename)
        if not os.path.exists(obj_path):
            print(f"Advertencia: No se encuentra {obj_path}. Saltando...")
            continue
            
        obj = o3d.io.read_point_cloud(obj_path)
        obj = obj.remove_non_finite_points()

        #ANADIDO POR EL TIEMPO
        start_time = time.time()

        # 2. Preprocesar el objeto (downsample, normales, FPFH)
        obj_down, obj_fpfh = preprocess_point_cloud(obj, VOXEL_SIZE, VECINDAD_NORMAL,VECINDAD_DESCRIPTORES,VECIN_NORM_MAX,VECIN_DESC_MAX)

        # 3. Alineamiento global con RANSAC
        result_ransac = execute_global_registration(obj_down, scene_down, obj_fpfh, scene_fpfh, VOXEL_SIZE, UMBRAL_RANSAC)
        
        # 4. Refinamiento con ICP
        result_icp = refine_registration(obj_down, scene_down, result_ransac.transformation, VOXEL_SIZE,UMBRAL_ICP)
        
        end_time = time.time()
        print(f"    -> Tiempo de PREPROCESADO + RANSAC + ICP: {end_time - start_time:.4f} segundos")
        print(f"    -> Fitness (Puntuación de encaje): {result_icp.fitness:.4f}")

        # 5. Transformar la nube original del objeto y pintarla
        obj_matched = copy.deepcopy(obj)
        obj_matched.transform(result_icp.transformation)
        obj_matched.paint_uniform_color(colors[i % len(colors)])

        # Añadir el objeto alineado y pintado a la visualización global
        final_visualization += obj_matched

    print("\n=== 3. VISUALIZACIÓN FINAL ===")
    # Visualizamos la escena procesada (sin planos) junto con los objetos encontrados pintados
    o3d.visualization.draw_geometries([final_visualization], window_name="Reconocimiento de Objetos 3D", width=1024, height=768)
    #o3d.visualization.draw_geometries([scene], window_name="Reconocimiento de Objetos 3D", width=1024, height=768)
if __name__ == "__main__":
    main()


    #que pasa si elimino el detector de keypoints
