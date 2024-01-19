import cv2
import pyautogui
import torch
import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np
import pygame as pg


pyautogui.FAILSAFE = False

BLUE = (255, 0, 0)  # BGR
GREEN = (0, 255, 0)
LINE_THICKNESS = 5
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# MIDAS
# https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-01-08-depth-vision-midas/2023-01-08/#choosing-the-right-model


MIDAS_SMALL = "MiDaS_small"  # (highest accuracy, slowest inference speed)
MIDAS_MEDIUM = "DPT_Hybrid"  # (medium accuracy, medium inference speed)
MIDAS_LARGE = "DPT_Large"  # (lowest accuracy, highest inference speed)
    
    

    




# ================
# ===> MIDAS <====
# ================

# Load Model
midas_model = MIDAS_LARGE
midas = torch.hub.load("intel-isl/MiDaS", str(midas_model))

# Choose Device (to run prediction on: CUDA/Nvidia vs. CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
midas.to(device)
midas.eval()

# Choose Transformation Pipeline
#   - inputs need to be 'transformed' to match the dataset format the model was trained with.
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if (midas_model == MIDAS_LARGE or midas_model == MIDAS_MEDIUM):
    transform = transforms.dpt_transform
else:
    transform = transforms.small_transform

# Calculate Prediction
input_image_dir_path = './input_images'
image_filename = os.listdir(input_image_dir_path)[0]
input_image = cv2.imread(os.path.join(input_image_dir_path, image_filename))

image_midas = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

imgbatch = transform(image_midas).to(device)

with torch.no_grad():
    prediction = midas(imgbatch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image_midas.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
output = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

##--- new

def conv_cv_alpha(cv_image, mask):    
    b, g, r = cv2.split(cv_image)    
    rgba = [r, g, b, mask]    
    cv_image = cv2.merge(rgba,4)    
          
    return cv_image


layers = []     
prev_thres = 255
div=30
     
for thres in range(255 - div, 0, -div):        
   ret, mask = cv2.threshold(output, thres, 255, cv2.THRESH_BINARY)
        
   ret, prev_mask = cv2.threshold(output, prev_thres, 255, cv2.THRESH_BINARY)  
       
   prev_thres = thres        
   inpaint_img = cv2.inpaint(input_image, prev_mask, 10, cv2.INPAINT_NS)
   layer = cv2.bitwise_and(inpaint_img, inpaint_img, mask = mask)    
   layers.append(conv_cv_alpha(layer, mask))  
    
# adding last layer 
   
mask = np.zeros(output.shape, np.uint8)    
mask[:,:] = 255   
 
ret, prev_mask = cv2.threshold(output, prev_thres, 255, cv2.THRESH_BINARY)
     
inpaint_img = cv2.inpaint(input_image, prev_mask, 10, cv2.INPAINT_NS)
layer = cv2.bitwise_and(inpaint_img, inpaint_img, mask = mask)
layers.append(conv_cv_alpha(layer, mask))
     
layers = layers[::-1]


face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')   

def get_face_rect(img):    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    face_rects = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    if len(face_rects) == 0:         
        return ()    
    return face_rects[0]

scale = 1
off_set = 20
width, height = layers[0].get_width(), layers[0].get_height()        
win = pg.display.set_mode((int((width - off_set)*scale), int((height - off_set)*scale)))    
pg.display.set_caption('Parallax_image')
scaled_layers = []    
for layer in layers: 
    scaled_layers.append(pg.transform.scale(layer, (int(width*scale), int(height*scale))))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


x_transform = True     # allow shift in x-axis
y_transform = False    # allow shift in y-axis
sens = 50              # the amount of scale down of shift value
show_cam = False       # show your face cam
shift_x = 0    
shift_y = 0    
run = True


while run:
    for event in pg.event.get():
        if event.type==pg.QUIT:
            run = False    
            ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    initial_pos = (frame.shape[0]/2, frame.shape[1]/2)
    face_rect = get_face_rect(frame)    
    if len(face_rect) != 0:
        x,y,w,h, = face_rect
        face_rect_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 3)        
        shift_x = (initial_pos[0] - (x + w/2))/(sens*scale)
        shift_y = (initial_pos[1] - (y + h/2))/(sens*scale)    
        win.fill((255, 255, 255))
                 
    for i, layer in enumerate(scaled_layers):
        new_x = -off_set/2
        new_y = -off_set/2
        if x_transform:
            new_x = 0 + shift_x*i
        if y_transform:
            new_y = 0 + shift_y*i
        win.blit(layer, (new_x, new_y)) 
        
    face_rect_frame = cv2.resize(face_rect_frame, (100, 100))
    if show_cam:
        win.blit(conv_cv_pygame(face_rect_frame), (0, 0))
    pg.display.update()

cap.release()
cv2.destroyAllWindows()
pg.quit()









# # Draw Depth Map
# output_cv2_display = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# cv2.imshow('Depth Map', output_cv2_display)

# cv2.imwrite('./rgbd_images/colour_img.jpg', input_image)
# cv2.imwrite('./rgbd_images/depth_img.jpg', output)


# # ======================
# # ===> Point Cloud <====
# # ======================
# # https://www.youtube.com/watch?v=vGr8Bg2Fda8
# # https://towardsdatascience.com/generate-a-3d-mesh-from-an-image-with-python-12210c73e5cc

# colour_raw = o3d.io.read_image('./rgbd_images/colour_img.jpg').flip_horizontal()
# depth_raw = o3d.io.read_image('./rgbd_images/depth_img.jpg').flip_horizontal()

# image_height, image_width, image_channels = cv2.imread('./rgbd_images/colour_img.jpg').shape

# # Obtain RGBD Image (Red, Green, Blue, Depth)
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(colour_raw, depth_raw)

# # Set Camera Settings (of input picture)
# camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
# camera_intrinsic.set_intrinsics(image_width, image_height, 500, 500, image_width/2, image_height/2)

# # Create Point Cloud
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# # o3d.visualization.draw_geometries([pcd])


# # ==========================
# # ===> Mesh Generation <====
# # ==========================

# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
# pcd = pcd.select_by_index(ind)

# # estimate normals
# pcd.estimate_normals()
# pcd.orient_normals_to_align_with_direction()

# # surface reconstruction
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# # rotate the mesh
# rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
# mesh.rotate(rotation, center=(0, 0, 0))

# # save the mesh
# o3d.io.write_triangle_mesh(f'./mesh.obj', mesh)

# # visualize the mesh
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


# ==========================
# ====> Colour Mapping <====
# ==========================
# Colour and depth images are not perfectly aligned so must colour map







# cap = cv2.VideoCapture(0)


# class Vector2D:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y


# screen_width, screen_height = pyautogui.size()

# eye_pos = Vector2D(screen_width / 2, screen_height / 2)
# count = 0

# while True:
#     ret, frame = cap.read()
#     camera_width = int(cap.get(3))
#     camera_height = int(cap.get(4))

    

    # ==== MiDaS ====
    # img_midas_camera = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #target camera
    
    # img_midas = img_midas_png
    # imgbatch = transform(img_midas).to(device)
    # # make prediction
    # with torch.no_grad():
    #     prediction = midas(imgbatch)
    #     prediction = torch.nn.functional.interpolate(
    #         prediction.unsqueeze(1),
    #         size=img_midas.shape[:2],
    #         mode="bicubic",
    #         align_corners=False,
    #     ).squeeze()
    #     output = prediction.cpu().numpy()
    # plt.imshow(output)
    # plt.pause(0.00001)

    # # Face Tracking
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    # if len(faces) > 0:
    #     x, y, w, h = faces[0]

    #     cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, LINE_THICKNESS)
    #     cv2.circle(frame, (x + int(w / 2), y + int(h / 2)), 5, GREEN, -1)

    #     eye_pos.x = x
    #     eye_pos.y = y

    # cv2.imshow("frame", frame)

    # # Mouse
    # eye_xposition_ratio = eye_pos.x / camera_width
    # eye_yposition_ratio = eye_pos.y / camera_height

    # target_mouse_x = screen_width - (screen_width * eye_xposition_ratio)
    # target_mouse_y = screen_height * eye_yposition_ratio
    # # pyautogui.moveTo(target_mouse_x, target_mouse_y)
    # count += 1

    # if count > 200:
    #     break

#     if cv2.waitKey(1) == ord("q"):
#         break


# cap.release()
# cv2.destroyAllWindows()
# plt.show()
