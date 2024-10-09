import copy
import open3d as o3d
import numpy as np

def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press q to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def manual_registration(src_pcd, dst_pcd):
    print("Demo for manual ICP")
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(dst_pcd, src_pcd, np.identity(4))

    ### pick points from two point clouds and build correspondences
    picked_id_source = []
    picked_id_target = []

    while True:
        picked_id_source += pick_points(src_pcd)
        picked_id_target += pick_points(dst_pcd)

        answer = input("Continue (y/n): ")
        if answer.lower() != "y":
            break

    print(picked_id_source)
    print(picked_id_target)
    assert len(picked_id_source) >= 3 and len(picked_id_target) >= 3
    assert len(picked_id_source) == len(picked_id_target)

    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # Estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(src_pcd, dst_pcd, o3d.utility.Vector2iVector(corr))

    # Point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.001  # 1mm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(src_pcd, dst_pcd, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())

    draw_registration_result(src_pcd, dst_pcd, reg_p2p.transformation)
    return reg_p2p.transformation

def load_pcd(file_path):
    """Load a PCD file into an Open3D PointCloud object."""
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"Failed to load point cloud from file: {file_path}")
    return pcd

def main():
    # File paths to the PCD files
    src_pcd_file = r'/test/map/map1.pcd'
    dst_pcd_file = r'/test/map/map2.pcd'

    # Load point clouds from PCD files
    src_pcd = load_pcd(src_pcd_file)
    dst_pcd = load_pcd(dst_pcd_file)

    # Perform manual registration
    transformation = manual_registration(src_pcd, dst_pcd)
    print("Estimated Transformation Matrix:")
    print(transformation)

if __name__ == "__main__":
    main()
