import argparse
import cv2
import pygame as pg
import os

from depth_map import create_depth_map, draw_depth_map
from model import create_point_cloud, draw_point_cloud, create_mesh, draw_mesh
from layered_parallax import create_parallax_layers_images, get_parallax_layers_image, draw_parallax_image, cv2pygame_image
from utils import Vector2D
from camera import Camera
from file_query import input_exists, read_image_metadata, write_image_metadata, delete_all_data

SCALE = 1 # scale of screen
BORDERS = 20  

class App:
    def __init__(self):
        pg.init()
        self.main_loop()

    def main_loop(self):
        args = self.parse_arguments()
        
        if (args.deleteall == True):
            delete_all_data()

        filename = args.input
        file_type = args.intype
        
        filename_without_extension = filename.split('.')[0]
        image_output_folder_path = os.path.join('output', 'images', filename_without_extension)
        
        # Check if image/video has already been processed
        if (input_exists(filename, file_type)):
            print('Input already computed.\nNow generating visualizer.')
        
        else:  # create image data
            # Make the folder name same as input filename
            if not os.path.exists(image_output_folder_path):
                os.makedirs(image_output_folder_path)
                os.makedirs(os.path.join(image_output_folder_path, 'depth_map'))
                os.makedirs(os.path.join(image_output_folder_path, 'layers'))
            else:
                print("ERROR: Folder already exists but shouldn't")
                return

            create_depth_map(filename, file_type, image_output_folder_path)
            
            layer_names = create_parallax_layers_images(filename, file_type, image_output_folder_path)
            
            write_image_metadata(filename, layer_names)


        src_data = read_image_metadata(filename)
        
        parallax_image_layers = get_parallax_layers_image(image_output_folder_path, src_data['parallax_layers'])

        width, height = parallax_image_layers[0].shape[1], parallax_image_layers[0].shape[0]
        

        # Create Pygame window
        window = pg.display.set_mode(
            (int( (width - BORDERS)*SCALE ), int( (height - BORDERS)*SCALE ))
        )
        pg.display.set_caption('Parallax Image')
        
        # Convert frame image layers to pygame layers
        parallax_pygame_layers = []
        for layer in parallax_image_layers:
            parallax_pygame_layers.append(
                pg.transform.scale(
                    cv2pygame_image(layer, mode='RGBA'), 
                    (
                        int( width*SCALE ), 
                        int( height*SCALE )
                    )
                )
            )


        # Setup controller
        if args.controller == 'face':
            cam = Camera()
            center_of_moveable_space = Vector2D(cam.dimensions.width/2, cam.dimensions.height/2)
        elif args.controller == 'mouse':
            cam = None
            center_of_moveable_space = Vector2D(width/2, height/2)


        # Main Loop
        running = True
        while running:
            # Get position of point of interest/controller (eyes or mouse)
            if args.controller == 'face':
                cam.update_frame()
                eyes = cam.track_eye()
                cam.draw_eye_tracking(eyes)
                x, y, w, h = eyes
                target_pos = (x + (w/2), y + (h/2))

            elif args.controller == 'mouse':
                x, y = pg.mouse.get_pos()
                target_pos = Vector2D(x, y)
                
            # Display Parallax Image(s)
            draw_parallax_image(window, parallax_pygame_layers, target_pos, center_of_moveable_space, args, SCALE, BORDERS)
            
            # Exit Methods (close pygame window or 'q' on camera)
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
            if cv2.waitKey(1) == ord('q'):
                running = False

        self.quit(cam)       
                
    def quit(self, cam):
        pg.quit()
        if cam is not None: del cam
        cv2.destroyAllWindows()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Specify process settings")
        parser.add_argument('--input', type=str, required=True)
        parser.add_argument('--intype', type=str, choices=['image', 'video'], required=True)
        parser.add_argument("--controller", type=str, choices=['face', 'mouse'])
        parser.add_argument("-deleteall", action='store_true')
        
        return parser.parse_args()


if __name__ == "__main__":
    app = App()
