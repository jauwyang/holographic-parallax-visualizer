import argparse
import cv2
import pygame as pg

from depth_map import create_depth_map, draw_depth_map
from model import create_point_cloud, draw_point_cloud, create_mesh, draw_mesh
from layered_parallax import create_parallax_layers, draw_parallax_image, cv2pygame_image
from utils import Vector2D
from camera import Camera

SCALE = 1 # scale of screen
BORDERS = 20  

class App:
    def __init__(self):
        pg.init()
        
        self.main_loop()

    def main_loop(self):
        args = self.parse_arguments()
    
        create_depth_map(args.model)
        # draw_depth_map()
        
        if args.view == '3d':
            create_point_cloud()
            # draw_point_cloud()
            create_mesh()
            # draw_mesh()
        elif args.view == '2d':
            parallax_layers = create_parallax_layers()

            width, height = parallax_layers[0].shape[1], parallax_layers[0].shape[0]
            window = pg.display.set_mode(
                (int( (width - BORDERS)*SCALE ), int( (height - BORDERS)*SCALE ))
            )
            pg.display.set_caption('Parallax Image')
            
            scaled_parallax_layers = []
            for layer in parallax_layers:
                scaled_parallax_layers.append(
                    pg.transform.scale(
                        cv2pygame_image(layer, mode='RGBA'), 
                        (
                            int( width*SCALE ), 
                            int( height*SCALE )
                        )
                    )
                )
        else:
            print("Invalid view")
            return
        
        cam = None
        if args.controller == 'face':
            cam = Camera()
            center_of_moveable_space = Vector2D(cam.dimensions.width/2, cam.dimensions.height/2)
        elif args.controller == 'mouse':
            center_of_moveable_space = Vector2D(width/2, height/2)
        
        running = True
        while running:
            if args.controller == 'face':
                cam.update_frame()
                eyes = cam.track_eye()
                cam.draw_eye_tracking(eyes)
                
                x, y, w, h = eyes
                target_pos = (x + (w/2), y + (h/2))
            elif args.controller == 'mouse':
                x, y = pg.mouse.get_pos()
                target_pos = Vector2D(x, y)
            
            if args.view == '3d':
                pass
            
            elif args.view == '2d':
                for event in pg.event.get():
                    if (event.type == pg.QUIT):
                        running = False
                        
                draw_parallax_image(window, scaled_parallax_layers, target_pos, center_of_moveable_space, args, SCALE, BORDERS)
            
            if cv2.waitKey(1) == ord('q'):
                running = False
                
        self.quit(cam)
                
    def quit(self, cam):
        pg.quit()
        if cam is not None: del cam
        cv2.destroyAllWindows()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Specify process settings")
        parser.add_argument("--model", type=str, choices=["large", "medium", "small"])
        parser.add_argument("--view", type=str, choices=['3d', '2d'])
        parser.add_argument("--controller", type=str, choices=['face', 'mouse'])
        
        return parser.parse_args()


if __name__ == "__main__":
    app = App()
