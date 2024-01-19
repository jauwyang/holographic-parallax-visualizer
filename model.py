import open3d as o3d
import cv2
import numpy as np

def create_point_cloud():
    # https://www.youtube.com/watch?v=vGr8Bg2Fda8
    # https://towardsdatascience.com/generate-a-3d-mesh-from-an-image-with-python-12210c73e5cc
    
    colour_raw = o3d.io.read_image('./rgbd_images/colour_img.jpg').flip_horizontal()
    depth_raw = o3d.io.read_image('./rgbd_images/depth_img.jpg').flip_horizontal()
    
    image_height, image_width, image_channels = cv2.imread('./rgbd_images/colour_img.jpg').shape

    # Obtain RGBD Image (Red, Green, Blue, Depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(colour_raw, depth_raw)

    # Set Camera Settings (of input picture)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(image_width, image_height, 500, 500, image_width/2, image_height/2)

    # Create Point Cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    o3d.io.write_point_cloud(f'./models/point_clouds/point_cloud.pcd', pcd)
    
    # return pcd


def draw_point_cloud():
    point_cloud = o3d.io.read_point_cloud('./models/point_clouds/point_cloud.pcd')
    o3d.visualization.draw_geometries([point_cloud])


def create_mesh():
    point_cloud = o3d.io.read_point_cloud('./models/point_clouds/point_cloud.pcd')
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    point_cloud = point_cloud.select_by_index(ind)

    # estimate normals
    point_cloud.estimate_normals()
    point_cloud.orient_normals_to_align_with_direction()

    # surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=10, n_threads=1)[0]

    # rotate the mesh
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))

    # save the mesh
    o3d.io.write_triangle_mesh(f'./models/mesh/mesh.obj', mesh)
    

def draw_mesh():
    # is upside down
    # has no colour
    mesh = o3d.io.read_triangle_mesh('./models/mesh/mesh.obj')
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)