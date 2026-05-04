import open3d as o3d
import numpy as np

# ─── Configuración central ───────────────────────────────────────────────────
VOXEL_SIZE_SCENE  = 0.005
VOXEL_SIZE_OBJECT = 0.001
FPFH_RADIUS       = 0.015
NORMAL_RADIUS     = 0.01

SCENE_PATH = "clouds/scenes/snap_0point.pcd"

OBJECT_PATHS = {
    "mug":       "clouds/objects/s0_mug_corr.pcd",
    "piggybank": "clouds/objects/s0_piggybank_corr.pcd",
    "plant":     "clouds/objects/s0_plant_corr.pcd",
    "plc":       "clouds/objects/s0_plc_corr.pcd",
}

# ─── Pasos 1-4 (igual que antes) ─────────────────────────────────────────────
def preprocess_scene(pcd_path, voxel_size):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"[scene | load]  {len(pcd.points):,} puntos originales")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[scene | voxel] {len(pcd_down.points):,} puntos tras downsampling")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=30))
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
            radius=NORMAL_RADIUS, max_nn=30))
    print(f"[{name:10} | clean] {len(pcd_down.points):,} puntos útiles\n")
    return pcd_down

def extract_keypoints(name, pcd, voxel_size):
    keypoints = pcd.uniform_down_sample(every_k_points=5)
    print(f"[{name:10} | uniform] {len(keypoints.points):,} keypoints")
    return keypoints

def compute_fpfh(name, keypoints, fpfh_radius):
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        keypoints,
        o3d.geometry.KDTreeSearchParamRadius(radius=fpfh_radius)
    )
    print(f"[{name:10} | fpfh] shape={np.array(fpfh.data).shape}  "
          f"radio={fpfh_radius*1000:.1f}mm")
    return fpfh

def compute_correspondences(name, obj_kp, obj_fpfh, scene_kp, scene_fpfh):
    obj_desc   = np.array(obj_fpfh.data).T
    scene_desc = np.array(scene_fpfh.data).T
    scene_tree = o3d.geometry.KDTreeFlann(scene_fpfh)
    obj_tree   = o3d.geometry.KDTreeFlann(obj_fpfh)
    correspondences = []
    for i in range(len(obj_desc)):
        _, idx_in_scene, _ = scene_tree.search_knn_vector_xd(obj_desc[i], 1)
        j = idx_in_scene[0]
        _, idx_in_obj, _ = obj_tree.search_knn_vector_xd(scene_desc[j], 1)
        if idx_in_obj[0] == i:
            correspondences.append([i, j])
    corr = np.array(correspondences) if correspondences else np.empty((0, 2), dtype=int)
    print(f"[{name:10} | corr] {len(corr)} correspondencias mutuas "
          f"(de {len(obj_desc)} keypoints del objeto)")
    return corr



# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # --- Pasos 1-2 ---
    scene_clean = preprocess_scene(SCENE_PATH, VOXEL_SIZE_SCENE)
    objects_clean = {name: preprocess_object(name, path, VOXEL_SIZE_OBJECT)
                     for name, path in OBJECT_PATHS.items()}

    print("─" * 50)
    scene_kp = extract_keypoints("scene", scene_clean, VOXEL_SIZE_SCENE)
    objects_kp = {name: extract_keypoints(name, pcd, VOXEL_SIZE_OBJECT)
                  for name, pcd in objects_clean.items()}

    # --- Paso 3 ---
    print("─" * 50)
    scene_fpfh   = compute_fpfh("scene", scene_kp, FPFH_RADIUS)
    objects_fpfh = {name: compute_fpfh(name, objects_kp[name], FPFH_RADIUS)
                    for name in OBJECT_PATHS}

    # --- Paso 4 ---
    print("─" * 50)
    correspondences = {}
    for name in OBJECT_PATHS:
        corr = compute_correspondences(
            name,
            objects_kp[name], objects_fpfh[name],
            scene_kp, scene_fpfh
        )
        correspondences[name] = corr