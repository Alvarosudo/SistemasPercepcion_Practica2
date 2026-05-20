import open3d as o3d
import numpy as np
import copy
import os
import time
import matplotlib.pyplot as plt

# =================================================================
# PARÁMETROS DE INVESTIGACIÓN (Configuración optimizada)
# =================================================================
VOXEL_SIZE = 0.005 

VECINDAD_NORMAL = VOXEL_SIZE * 2 
VECIN_NORM_MAX = 30 

VECINDAD_DESCRIPTORES = VOXEL_SIZE * 5 
VECIN_DESC_MAX = 100 

SAMPLES_RANSAC = 3 
UMBRAL_RANSAC = VOXEL_SIZE * 1.5 
UMBRAL_ICP = VOXEL_SIZE * 1.5

# Configuración de rutas
SCENE_PATH = "clouds/scenes/snap_0point.pcd"
OBJECTS_DIR = "clouds/objects/"
OBJECT_FILES = ["s0_mug_corr.pcd", "s0_plc_corr.pcd", "s0_plant_corr.pcd", "s0_piggybank_corr.pcd"]

# =================================================================
# FUNCIONES DE PROCESADO Y REGISTRO
# =================================================================

def remove_planes(pcd):
    """ Segmenta y elimina el plano dominante de la mesa """
    _, inliers = pcd.segment_plane(
        distance_threshold=0.015,
        ransac_n=SAMPLES_RANSAC,
        num_iterations=1000
    )
    return pcd.select_by_index(inliers, invert=True)

def preprocess_point_cloud(pcd):
    """ Downsampling, extracción de KEYPOINTS (ISS) y Descriptor Slicing """
    # 1. Filtrar la nube para reducir su tamaño (Downsampling)
    pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VECINDAD_NORMAL, max_nn=VECIN_NORM_MAX)
    )

    # 2. CALCULAR DESCRIPTORES EN LA NUBE DENSA
    dense_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=VECINDAD_DESCRIPTORES, max_nn=VECIN_DESC_MAX)
    )

    # 3. Extraer puntos característicos (ISS)
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_down,
        salient_radius=VOXEL_SIZE * 5,
        non_max_radius=VOXEL_SIZE * 2,
        gamma_21=0.99,
        gamma_32=0.99,
        min_neighbors=3
    )

    # 4. CONTROL DE SEGURIDAD (Fallback adaptativo)
    usando_fallback = False
    if len(keypoints.points) < 25: # Si el objeto es muy liso y plano
        keypoints = pcd_down
        kp_fpfh = dense_fpfh
        usando_fallback = True
    else:
        # 5. DESCRIPTOR SLICING (Mapeo de descriptores ricos)
        kdtree = o3d.geometry.KDTreeFlann(pcd_down)
        kp_indices = []
        for pt in keypoints.points:
            _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
            kp_indices.append(idx[0])
        
        kp_fpfh = o3d.pipelines.registration.Feature()
        kp_fpfh.data = dense_fpfh.data[:, kp_indices]

    # Recalcular normales de los keypoints para RANSAC
    keypoints.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VECINDAD_NORMAL, max_nn=VECIN_NORM_MAX)
    )
    
    # Devolvemos también dense_fpfh para permitir el emparejamiento simétrico
    return pcd_down, keypoints, kp_fpfh, dense_fpfh, usando_fallback

def execute_global_registration(source_kp, target_kp, source_fpfh, target_fpfh):
    """ Alineación global mediante RANSAC """
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_kp, target_kp, source_fpfh, target_fpfh, True, UMBRAL_RANSAC,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        SAMPLES_RANSAC, 
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(UMBRAL_RANSAC)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

def refine_registration(source_down, target_down, initial_transform):
    """ Refinamiento local mediante ICP Point-to-Plane """
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, UMBRAL_ICP, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

# =================================================================
# PIPELINE PRINCIPAL
# =================================================================

def main():
    print("=== 1. CARGANDO Y PREPROCESANDO LA ESCENA ===")
    scene = o3d.io.read_point_cloud(SCENE_PATH).remove_non_finite_points()
    
    print("-> Eliminando plano dominante de la escena...")
    scene_no_plane = remove_planes(scene)
    
    print("-> Extrayendo Keypoints y Descriptores de la escena...")
    # Obtenemos tanto la versión ISS como la versión Densa de la escena
    scene_down, scene_kp, scene_fpfh, scene_dense_fpfh, _ = preprocess_point_cloud(scene_no_plane)
    print(f"   [!] Se encontraron {len(scene_kp.points)} Keypoints en la escena.")

    final_visualization = copy.deepcopy(scene)
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

    nombres_objetos = []
    lista_fitness = []
    lista_rmse = []
    lista_tiempos = []

    print("\n=== 2. PROCESANDO OBJETOS Y BUSCÁNDOLOS EN LA ESCENA ===")
    print(f"\n{'OBJETO':<15} | {'KEYPOINTS':<9} | {'MÉTODO':<10} | {'FITNESS':<8} | {'RMSE (m)':<10}")
    print("-" * 70)
    
    for i, obj_filename in enumerate(OBJECT_FILES):
        obj_path = os.path.join(OBJECTS_DIR, obj_filename)
        if not os.path.exists(obj_path):
            continue
            
        obj = o3d.io.read_point_cloud(obj_path).remove_non_finite_points()
        start_time = time.time()
        
        # Preprocesar objeto
        obj_down, obj_kp, obj_fpfh, obj_dense_fpfh, usado_fallback = preprocess_point_cloud(obj)
        
        # ALINEAMIENTO GLOBAL RANSAC (Selección simétrica de la escena)
        if usado_fallback:
            modo_ejecucion = "Voxel (FB)"
            # Si el objeto es denso, lo buscamos en la escena densa
            result_ransac = execute_global_registration(obj_down, scene_down, obj_fpfh, scene_dense_fpfh)
        else:
            modo_ejecucion = "ISS+Rich"
            # Si el objeto es disperso (ISS), lo buscamos en la escena dispersa (ISS)
            result_ransac = execute_global_registration(obj_kp, scene_kp, obj_fpfh, scene_fpfh)
        
        # Refinamiento ICP
        result_icp = refine_registration(obj_down, scene_down, result_ransac.transformation)
        duracion = time.time() - start_time
        
        nombre_corto = obj_filename.replace("s0_", "").replace("_corr.pcd", "").upper()
        num_kps = len(obj_kp.points)
        print(f"{nombre_corto:<15} | {num_kps:<9} | {modo_ejecucion:<10} | {result_icp.fitness:<8.4f} | {result_icp.inlier_rmse:<10.6f}")

        nombres_objetos.append(nombre_corto)
        lista_fitness.append(result_icp.fitness)
        lista_rmse.append(result_icp.inlier_rmse)
        lista_tiempos.append(duracion)

        obj_matched = copy.deepcopy(obj)
        obj_matched.transform(result_icp.transformation)
        obj_matched.paint_uniform_color(colors[i % len(colors)])
        final_visualization += obj_matched

    # Generación de gráficas
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Pipeline Adaptativo Simétrico (Resultados Finales)', fontsize=14, fontweight='bold')
    colores_barras = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f']

    axs[0].bar(nombres_objetos, lista_fitness, color=colores_barras, edgecolor='black')
    axs[0].set_title('Fitness (Puntuación)')
    axs[0].set_ylim(0, 1.05)
    for idx, v in enumerate(lista_fitness): axs[0].text(idx, v + 0.02, f"{v:.3f}", ha='center')

    axs[1].bar(nombres_objetos, lista_rmse, color=colores_barras, edgecolor='black')
    axs[1].set_title('RMSE')
    axs[1].set_ylim(0, max(lista_rmse) * 1.2 if lista_rmse else 0.02)
    for idx, v in enumerate(lista_rmse): axs[1].text(idx, v + (max(lista_rmse)*0.02), f"{v:.5f}", ha='center')

    axs[2].bar(nombres_objetos, lista_tiempos, color=colores_barras, edgecolor='black')
    axs[2].set_title('Tiempo de Ejecución')
    for idx, v in enumerate(lista_tiempos): axs[2].text(idx, v + (max(lista_tiempos)*0.02), f"{v:.3f}s", ha='center')

    plt.tight_layout()
    plt.savefig("graficas_resultados.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== 3. VISUALIZACIÓN FINAL ===")
    o3d.visualization.draw_geometries([final_visualization], window_name="Pipeline Avanzado de Registro 3D", width=1024, height=768)

if __name__ == "__main__":
    main()