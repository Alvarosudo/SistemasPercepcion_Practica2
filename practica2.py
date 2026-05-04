import open3d as o3d
import numpy as np
import copy
import os

def preprocess_point_cloud(pcd, voxel_size):
    """
    Realiza el filtrado (subsampling), cálculo de normales y cálculo de descriptores.
    """
    #(2 FIltrar para reducir tamano) 1. Filtrar la nube para reducir su tamaño (Voxel Downsampling)
    print(f"    -> Aplicando Voxel Downsampling con tamaño de voxel: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    #(3 Extraer puntos caracteristicos) 2. Extraer características (Cálculo de normales)
    radius_normal = voxel_size * 2
    print(f"    -> Calculando normales con radio de búsqueda: {radius_normal}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #(4 Calcular los descriptores para los puntos característicos) 3. Calcular descriptores (FPFH - Fast Point Feature Histograms)
    radius_feature = voxel_size * 5
    print(f"    -> Calculando descriptores FPFH con radio: {radius_feature}")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Alineación global inicial (Coarse alignment) usando RANSAC.
    """
    distance_threshold = voxel_size * 1.5
    print(f"    -> Ejecutando RANSAC (Alineamiento global). Umbral de distancia: {distance_threshold}")
    
    # RANSAC para alinear basándose en el emparejamiento de las características FPFH
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            # Comprobaciones para asegurar que los emparejamientos son lógicos geométricamente
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result

def refine_registration(source_down, target_down, initial_transform, voxel_size):
    """
    Refinamiento local (Fine alignment) usando ICP (Iterative Closest Point).
    """
    distance_threshold = voxel_size * 0.4
    print(f"    -> Ejecutando ICP (Refinamiento). Umbral de distancia: {distance_threshold}")
    
    # ICP (Point-to-Plane suele converger mejor y más rápido que Point-to-Point)
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return result

def main():
    # Rutas de los archivos (ajusta las rutas según dónde tengas los ficheros descomprimidos)
    SCENE_PATH = "clouds/scenes/snap_0point.pcd"
    OBJECTS_DIR = "clouds/objects/"
    
    # Parámetro clave: tamaño del voxel. Si los puntos están en metros, 0.005 son 5 milímetros.
    # ¡Prueba a cambiar este valor para tu memoria!
    VOXEL_SIZE = 0.005 

    print("=== 1. CARGANDO Y PREPROCESANDO LA ESCENA ===")
    scene = o3d.io.read_point_cloud(SCENE_PATH)
    
    # Eliminar NaNs e infinitos de la escena provenientes del escáner
    scene = scene.remove_non_finite_points()

    #(1 Eliminar todos los planos dominantes de la escena)
    print("-> Eliminando el plano dominante de la escena (RANSAC plano)")
    # distance_threshold: tolerancia de distancia al plano
    plane_model, inliers = scene.segment_plane(distance_threshold=0.015,
                                               ransac_n=3,
                                               num_iterations=1000)
    
    # Nos quedamos con la escena SIN el plano (invert=True)
    scene_no_plane = scene.select_by_index(inliers, invert=True)
    
    # Preprocesamos la escena (downsample, normales, FPFH)
    scene_down, scene_fpfh = preprocess_point_cloud(scene_no_plane, VOXEL_SIZE)

    # Nube final que usaremos para visualizar (una copia para no modificar la original)
    final_visualization = copy.deepcopy(scene)
    #final_visualization.paint_uniform_color([0.8, 0.8, 0.8]) # Escena en gris claro

    # Lista de archivos de objetos a buscar
    object_files = ["s0_mug_corr.pcd", "s0_plc_corr.pcd", "s0_plant_corr.pcd", "s0_piggybank_corr.pcd"]
    
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

        # 2. Preprocesar el objeto (downsample, normales, FPFH)
        obj_down, obj_fpfh = preprocess_point_cloud(obj, VOXEL_SIZE)

        # 3. Alineamiento global con RANSAC
        result_ransac = execute_global_registration(obj_down, scene_down, obj_fpfh, scene_fpfh, VOXEL_SIZE)
        
        # 4. Refinamiento con ICP
        result_icp = refine_registration(obj_down, scene_down, result_ransac.transformation, VOXEL_SIZE)
        
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

if __name__ == "__main__":
    main()
