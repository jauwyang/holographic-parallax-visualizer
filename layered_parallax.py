import cv2
import numpy as np
import pygame as pg
import os

from config import PARALLAX_SENSITIVITY, X_TRANSFORM, Y_TRANSFORM, DIVISION_SIZE

WHITE = (255, 255, 255)

# https://medium.com/analytics-vidhya/parallax-images-14e92ebb1bae
# https://github.com/strikeraryu/Parallax_Image/blob/19a48f93e695fed3635f5018d0ed4a5c4b6c4cdd/Parallax_Image/parallax_image.py#L52

def cv2alpha_image(cv_image, mask):
    """ 
    Adds alpha value to pixels for transparency
    """
    b, g, r = cv2.split(cv_image)
    rgba = [r, g, b, mask]
    cv_image = cv2.merge(rgba, 4)
    
    return cv_image

def cv2pygame_image(cv_image, mode='RGBA'):
    size = cv_image.shape[1::-1]
    data = cv_image.tobytes()
    frame_pg = pg.image.fromstring(data, size, mode)

    return frame_pg


def create_parallax_layers(img_src, image_output_folder_path):
    depth_map = cv2.imread(os.path.join(image_output_folder_path, 'depth_map', 'depth_map.png'))
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img_src, depth_map.shape[::-1])
    
    layers = []
    prev_threshold = 255
    
    for curr_threshold in range(255-DIVISION_SIZE, 0, -DIVISION_SIZE):
        ret, curr_mask = cv2.threshold(depth_map, curr_threshold, 255, cv2.THRESH_BINARY)
        ret, prev_mask = cv2.threshold(depth_map, prev_threshold, 255, cv2.THRESH_BINARY)
        
        prev_threshold = curr_threshold
        
        # inpaint the previous layer ONLY with the content infront of it**
        inpaint_image = cv2.inpaint(img, prev_mask, 10, cv2.INPAINT_NS)
        
        layer = cv2.bitwise_and(inpaint_image, inpaint_image, mask=curr_mask)
        
        layers.append(cv2alpha_image(layer, mask=curr_mask))
    
    # add last layer (front) of image
    mask = np.zeros(depth_map.shape, np.uint8)
    mask[:,:] = 255
    ret, prev_mask = cv2.threshold(depth_map, prev_threshold, 255, cv2.THRESH_BINARY)
    inpaint_img = cv2.inpaint(img, prev_mask, 10, cv2.INPAINT_NS)
    layer = cv2.bitwise_and(inpaint_img, inpaint_img, mask = mask)
    layer_image = cv2alpha_image(layer, mask)
    layers.append(layer_image)
    
    layers = layers[::-1]  # reverse order (background is first, foreground last)
    layer_names = []
    
    for i, layer in enumerate(layers):
        file_layer_name = str(i).zfill(3) + '.png'
        layer_names.append(file_layer_name)   # create layer names for JSON metadata
        cv2.imwrite(os.path.join(image_output_folder_path, 'layers', file_layer_name), layer)

    return layer_names


def get_parallax_layers(image_output_folder_path, layer_names):
    folder_path = os.path.join(image_output_folder_path, 'layers')
    layer_images = []
    for layer_name in layer_names:
        layer = cv2.imread(os.path.join(folder_path, layer_name), cv2.IMREAD_UNCHANGED)  # opencv removes alpha (transparency channel when reading so make sure to keep)
        layer_images.append(layer)
    
    return layer_images



def draw_parallax_frame(window, layers, target_pos, center_of_moveable_space, args, scale=1, offset=20):    
    shift_x = 0
    shift_y = 0
    
    if (target_pos is not None):
        shift_x = (center_of_moveable_space.x - target_pos.x)/(PARALLAX_SENSITIVITY * scale)
        shift_y = (center_of_moveable_space.y - target_pos.y)/(PARALLAX_SENSITIVITY * scale)
    
    # Draw Frame    
    window.fill(WHITE)
    flip_direction = 1
    if args.controller == 'face':
        flip_direction = -1
    for i, layer in enumerate(layers):
        new_x = -offset/2
        new_y = -offset/2
        
        if X_TRANSFORM:
            new_x = shift_x*i
        if Y_TRANSFORM:
            new_y = shift_y*i
        
        window.blit(layer, (flip_direction * new_x, new_y))

    pg.display.update()
