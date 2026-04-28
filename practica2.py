import open3d as o3d
import numpy as np

# ─── Configuración central ───────────────────────────────────────────────────
VOXEL_SIZE_SCENE  = 0.005
VOXEL_SIZE_OBJECT = 0.001

SCENE_PATH = "clouds/scenes/snap_0point.pcd"

OBJECT_PATHS = {
    "mug":       "clouds/objects/s0_mug_corr.pcd",
    "piggybank": "clouds/objects/s0_piggybank_corr.pcd",
    "plant":     "clouds/objects/s0_plant_corr.pcd",
    "plc":       "clouds/objects/s0_plc_corr.pcd",
}

# ─── Pasos 1 y 2 (igual que antes) ───────────────────────────────────────────
def preprocess_scene(pcd_path, voxel_size):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"[scene | load]  {len(pcd.points):,} puntos originales")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[scene | voxel] {len(pcd_down.points):,} puntos tras downsampling")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30))
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"[scene | plane] {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f"[scene | plane] {len(inliers):,} puntos eliminados")
    pcd_clean = pcd_down.select_by_index(inliers, invert=True)
    print(f"[scene | clean] {len(pcd_clean.points):,} puntos útiles\n")
    return pcd_clean

def preprocess_object(name, pcd_path, voxel_size):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"[{name:10} | load]  {len(pcd.points):,} puntos originales")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[{name:10} | voxel] {len(pcd_down.points):,} puntos tras downsampling")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30))
    print(f"[{name:10} | clean] {len(pcd_down.points):,} puntos útiles\n")
    return pcd_down

def extract_keypoints(name, pcd, voxel_size):
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=voxel_size * 6,
        non_max_radius=voxel_size * 4,
        gamma_21=0.975,
        gamma_32=0.975,
    )
    ratio = len(keypoints.points) / len(pcd.points) * 100
    print(f"[{name:10} | iss] {len(keypoints.points):,} keypoints "
          f"({ratio:.1f}% de {len(pcd.points):,})")
    return keypoints

# ─── Paso 3: Descriptores FPFH ────────────────────────────────────────────────
def compute_fpfh(name, pcd_full, keypoints, voxel_size):
    """
    Calcula descriptores FPFH solo sobre los keypoints,
    pero usando la nube completa para buscar vecinos.

    Por qué pcd_full y no solo keypoints:
      FPFH necesita vecinos reales alrededor de cada keypoint.
      Si solo le damos los keypoints, la vecindad queda vacía
      y el histograma sale plano e inútil.

    Args:
        pcd_full   : nube completa preprocesada (con normales)
        keypoints  : puntos clave extraídos por ISS
        voxel_size : para calcular el radio del descriptor

    Returns:
        fpfh : objeto Feature con matriz (33, N_keypoints)
    """
    # El radio debe ser > radio de normales (voxel_size*2)
    # Usamos voxel_size*5 como regla general segura
    radius_feature = voxel_size * 5

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        keypoints,
        o3d.geometry.KDTreeSearchParamRadius(radius=radius_feature)
    )

    # fpfh.data es una matriz numpy de shape (33, N_keypoints)
    print(f"[{name:10} | fpfh] shape={np.array(fpfh.data).shape}  "
          f"radio={radius_feature*1000:.1f}mm")

    return fpfh


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # --- Pasos 1 y 2 ---
    scene_clean = preprocess_scene(SCENE_PATH, VOXEL_SIZE_SCENE)
    objects_clean = {name: preprocess_object(name, path, VOXEL_SIZE_OBJECT)
                     for name, path in OBJECT_PATHS.items()}

    print("─" * 50)
    scene_kp = extract_keypoints("scene", scene_clean, VOXEL_SIZE_SCENE)
    objects_kp = {name: extract_keypoints(name, pcd, VOXEL_SIZE_OBJECT)
                  for name, pcd in objects_clean.items()}

    # --- Paso 3: FPFH ---
    print("─" * 50)
    scene_fpfh = compute_fpfh("scene", scene_clean, scene_kp, VOXEL_SIZE_SCENE)
    objects_fpfh = {name: compute_fpfh(name, objects_clean[name], objects_kp[name], VOXEL_SIZE_OBJECT)
                    for name in OBJECT_PATHS}
    
    print("\nVisualizando escena con keypoints (rojo)...")
    scene_kp_vis = o3d.geometry.PointCloud(scene_kp)
    scene_kp_vis.paint_uniform_color([1.0, 0.1, 0.1])
    o3d.visualization.draw_geometries(
        [scene_clean, scene_kp_vis],
        window_name="Escena — keypoints ISS"
    )