import open3d as o3d
import numpy as np
import copy
import os
import time

# =================================================================
# PARÁMETROS DE INVESTIGACIÓN (Configuración rápida)
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
    """ Downsampling, cálculo de normales y descriptores FPFH """
    pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VECINDAD_NORMAL, max_nn=VECIN_NORM_MAX)
    )

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=VECINDAD_DESCRIPTORES, max_nn=VECIN_DESC_MAX)
    )
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
    """ Alineación global rápida mediante RANSAC con filtrado geométrico """
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, UMBRAL_RANSAC,
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
    """ Refinamiento local ultra rápido mediante ICP Point-to-Plane """
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
    
    print("-> Eliminando plano de la mesa...")
    scene_no_plane = remove_planes(scene)
    
    print("-> Extrayendo características de la escena...")
    scene_down, scene_fpfh = preprocess_point_cloud(scene_no_plane)

    # Inicialización del contenedor para visualización final
    final_visualization = copy.deepcopy(scene)
    
    colors = [
        [1, 0, 0],    # Rojo (Mug)
        [0, 1, 0],    # Verde (PLC)
        [0, 0, 1],    # Azul (Plant)
        [1, 1, 0]     # Amarillo (Piggybank)
    ]

    print("\n=== 2. PROCESANDO OBJETOS Y BUSCÁNDOLOS EN LA ESCENA ===")
    print(f"\n{'OBJETO':<20} | {'FITNESS':<10} | {'RMSE':<12} | {'TIEMPO (s)':<10}")
    print("-" * 60)
    
    for i, obj_filename in enumerate(OBJECT_FILES):
        obj_path = os.path.join(OBJECTS_DIR, obj_filename)
        if not os.path.exists(obj_path):
            continue
            
        obj = o3d.io.read_point_cloud(obj_path).remove_non_finite_points()

        # Medición de tiempo del pipeline para este objeto
        start_time = time.time()

        # 1. Preprocesar objeto
        obj_down, obj_fpfh = preprocess_point_cloud(obj)

        # 2. Alineamiento global RANSAC
        result_ransac = execute_global_registration(obj_down, scene_down, obj_fpfh, scene_fpfh)
        
        # 3. Refinamiento ICP
        result_icp = refine_registration(obj_down, scene_down, result_ransac.transformation)
        
        duracion = time.time() - start_time
        
        # Impresión limpia en formato tabla
        print(f"{obj_filename:<20} | {result_icp.fitness:<10.4f} | {result_icp.inlier_rmse:<12.6f} | {duracion:<10.4f}")

        # Transformar, pintar e integrar en el modelo final
        obj_matched = copy.deepcopy(obj)
        obj_matched.transform(result_icp.transformation)
        obj_matched.paint_uniform_color(colors[i % len(colors)])
        
        final_visualization += obj_matched

    print("\n=== 3. VISUALIZACIÓN FINAL ===")
    o3d.visualization.draw_geometries(
        [final_visualization], 
        window_name="Reconocimiento Rápido de Objetos 3D", 
        width=1024, 
        height=768
    )

if __name__ == "__main__":
    main()