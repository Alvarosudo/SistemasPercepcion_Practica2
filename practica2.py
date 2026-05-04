import open3d as o3d
import numpy as np

# Configuración de rutas
SCENE_PATH = "clouds/scenes/snap_0point.pcd"
OBJECT_PATHS = {
    "mug":       "clouds/objects/s0_mug_corr.pcd",
    "piggybank": "clouds/objects/s0_piggybank_corr.pcd",
    "plant":     "clouds/objects/s0_plant_corr.pcd",
    "plc":       "clouds/objects/s0_plc_corr.pcd",
}

def remove_planes(pcd):
    """ Segmenta y elimina el plano de la mesa """
    _, inliers = pcd.segment_plane(0.01, 3, 1000)
    return pcd.select_by_index(inliers, invert=True)

def preprocess(pcd, voxel_size):
    """ Downsampling, normales y descriptores FPFH """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh

def execute_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """ Pipeline de registro: RANSAC + ICP Plane + ICP Point """
    
    # 1. Registro Global (RANSAC)
    dist_ransac = voxel_size * 1.5
    res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, dist_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_ransac)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    # 2. Refinamiento ICP Paso 1 (Point-to-Plane)
    res_plane = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size * 3, res_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # 3. Refinamiento ICP Paso 2 (Point-to-Point)
    res_final = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size * 1.5, res_plane.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return res_final

def main():
    voxel_size = 0.005
    
    # Cargar escena y preparar versión para registro
    scene = o3d.io.read_point_cloud(SCENE_PATH)
    scene_clean = remove_planes(scene)
    scene_down, scene_fpfh = preprocess(scene_clean, voxel_size)

    # Lista para visualización (empezamos con la escena original)
    visualizacion = [scene]

    for name, path in OBJECT_PATHS.items():
        print(f"\n--- Procesando: {name} ---")
        
        # Cargar y preparar objeto
        obj = o3d.io.read_point_cloud(path)
        obj_down, obj_fpfh = preprocess(obj, voxel_size)

        # Registro
        result = execute_registration(obj_down, scene_down, obj_fpfh, scene_fpfh, voxel_size)

        # Métricas
        eval_metrics = o3d.pipelines.registration.evaluate_registration(
            obj_down, scene_down, voxel_size * 1.5, result.transformation)
        print(f"  Fitness: {eval_metrics.fitness:.4f} | RMSE: {eval_metrics.inlier_rmse:.6f}")

        # Aplicar transformación al objeto original (mantiene sus colores originales)
        obj.transform(result.transformation)
        visualizacion.append(obj)

    print("\nAbriendo visor 3D ...")
    o3d.visualization.draw_geometries(visualizacion, window_name="Registro Final: Colores Originales")

if __name__ == "__main__":
    main()