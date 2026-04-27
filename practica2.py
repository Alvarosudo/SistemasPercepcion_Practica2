import open3d as o3d
import numpy as np

def preprocess(pcd_path, voxel_size):

    #1 Cargar la nube de puntos
    pcd = o3d.io.read_point_cloud (pcd_path)

    #2 Realizar voxelsampling para reducir su tamaño
    pcd_down = pcd.voxel_down_sample(voxel_size)

    #3 Eliminar el plano dominante
    plane_model, inliers= pcd_down.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    pcd_clean= pcd_down.select_by_index(inliers, invert=True)


    return pcd_down, pcd_clean

if __name__ == "__main__":

    scene_clean, scene_down = preprocess("clouds/scenes/snap_0point.pcd", voxel_size=0.001)
    obj_clean,   obj_down   = preprocess("clouds/objects/s0_mug_corr.pcd",  voxel_size=0.001)

    # Visualización opcional
     # naranja: objeto
    o3d.visualization.draw_geometries([scene_clean, obj_clean])