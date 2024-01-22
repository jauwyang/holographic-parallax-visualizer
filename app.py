import argparse
import cv2
import pygame as pg
import os

from depth_map import create_depth_map, draw_depth_map
from model import create_point_cloud, draw_point_cloud, create_mesh, draw_mesh
from layered_parallax import create_parallax_layers_images, get_parallax_layers_image, draw_parallax_image, cv2pygame_image
from utils import Vector2D
from camera import Camera
from file_query import input_image_exists, input_video_exists, read_image_metadata, write_image_metadata, read_video_metadata, write_video_metadata, delete_all_data
from video import video2frames


SCALE = 1 # scale of screen
BORDERS = 20  

class App:
    def __init__(self):
        pg.init()
        self.main_loop()

    def main_loop(self):
        args = self.parse_arguments()
        
        if (args.deleteall):
            delete_all_data()
            return

        filename = args.input
        file_type = args.intype
        
        filename_without_extension = filename.split('.')[0]
        
        # Check if image/video has already been processed
        if file_type == 'image':
            file_exists = input_image_exists(filename)
            image_output_folder_path = os.path.join('output', 'images', filename_without_extension)
        elif file_type == 'video':
            file_exists = input_video_exists(filename)
            image_output_folder_path = os.path.join('output', 'videos', filename_without_extension)
            # image_output_folder_path = './output/videos/<video_name>/
        else:
            print("File type doesn't exist")
            return
        
        # Process if image does not exisst
        if (file_exists):
            print('Input already computed.\nNow generating visualizer.')
        
        else:  # create image data
            # Make the folder name same as input filename
            if not os.path.exists(image_output_folder_path):
                os.makedirs(image_output_folder_path)
                if file_type == 'image':
                    input_path = os.path.join('input_src', filename)
                    os.makedirs(os.path.join(image_output_folder_path, 'layers'))
                    os.makedirs(os.path.join(image_output_folder_path, 'depth_map'))
                    
                    create_depth_map(input_path, file_type, image_output_folder_path)
            
                    layer_names = create_parallax_layers_images(input_path, file_type, image_output_folder_path)
                    
                    write_image_metadata(filename, layer_names)

                elif file_type == 'video':
                    video_all_frames_path = os.path.join(image_output_folder_path, 'frames')
                    os.makedirs(video_all_frames_path)  # creates parent 'frames' dir

                    video_frames = video2frames(os.path.join('input_src', filename))
                    frame_count = len(video_frames)
                    
                    
                    frames = {}
                    
                    for i, frame in enumerate(video_frames):
                        counter = str(i).zfill(4)
                        frame_folder_name = 'frame' + counter
                        single_frame_path = os.path.join(video_all_frames_path, frame_folder_name)
                        os.makedirs(single_frame_path) # creates single frame dir (e.g. './frame001/') <<< MAIN
                        
                        os.makedirs(os.path.join(single_frame_path, 'depth_map'))  # creates 'depth_map' dir for 1 frame (e.g. './frame001/depth_map/')
                        os.makedirs(os.path.join(single_frame_path, 'layers'))  # creates 'layers' dir for all layers for 1 frame (e.g. './frame001/layers/')
                        
                        # Create depth map for 1 frame
                        create_depth_map(frame, file_type, single_frame_path)
                        
                        # Create layers
                        layer_names = create_parallax_layers_images(frame, file_type, single_frame_path)
                        
                        frames['parallax_layers' + counter] = layer_names
                    
                    write_video_metadata(filename, frames, frame_count)
                    
            else:
                print("ERROR: Folder already exists but shouldn't")
                return
# output/
# | -- images/
# |    | -- image01/
# |    |    | -- depth_map/
# |    |    |    | -- depth_map.png
# |    |    | -- layers/
# |    |    |    | -- layer001.png
# |    |    |    | -- layer002.png
# |    |    |    | -- layer003.png
# |
# | -- videos/
# |    | -- video01/
# |    |    | -- frames/
# |    |    |    | -- frame001/
# |    |    |    |    | -- depth_map/
# |    |    |    |    |    | -- depth_map.png
# |    |    |    |    | -- layers/
# |    |    |    |    |    | -- layer001.png
# |    |    |    |    |    | -- layer002.png
# |    |    |    |    |    | -- layer003.png
# |    |    |    | -- frame002/
# |    |    |    |    | -- depth_map/
# |    |    |    |    |    | -- depth_map.png
# |    |    |    |    | -- layers/
# |    |    |    |    |    | -- layer001.png
# |    |    |    |    |    | -- layer002.png
# |    |    |    |    |    | -- layer003.png

        if (file_type == 'image'):
            src_data = read_image_metadata(filename)
        
            parallax_image_layers = get_parallax_layers_image(image_output_folder_path, src_data['parallax_layers'])

            width, height = parallax_image_layers[0].shape[1], parallax_image_layers[0].shape[0]
            
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
        
        elif (file_type == 'video'):
            src_data = read_video_metadata(filename)
            
            frames = []
            
            frame_count = src_data['frame_count']
            
            video_all_frames_path = os.path.join(image_output_folder_path, 'frames')
            
            for i in range(frame_count):
                counter = str(i).zfill(4)
                single_frame_path = os.path.join(video_all_frames_path, 'frame' + counter)
                parallax_video_frame_layers = get_parallax_layers_image(single_frame_path, src_data['frames']['parallax_layers' + counter])
                
                pygame_frame_layers = []
                width, height = parallax_video_frame_layers[0].shape[1], parallax_video_frame_layers[0].shape[0]
                
                for layer in parallax_video_frame_layers:
                    pygame_frame_layers.append(
                        pg.transform.scale(
                            cv2pygame_image(layer, mode='RGBA'), 
                            (
                                int( width*SCALE ), 
                                int( height*SCALE )
                            )
                        )
                    )
                
                frames.append(pygame_frame_layers)
        

        # Create Pygame window
        window = pg.display.set_mode(
            (int( (width - BORDERS)*SCALE ), int( (height - BORDERS)*SCALE ))
        )
        pg.display.set_caption('Parallax Image')


        # Setup controller
        if args.controller == 'face':
            cam = Camera()
            center_of_moveable_space = Vector2D(cam.dimensions.width/2, cam.dimensions.height/2)
        elif args.controller == 'mouse':
            cam = None
            center_of_moveable_space = Vector2D(width/2, height/2)


        # Main Loop
        running = True
        video_frame_iterator = 0
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
            if file_type == 'image':
                draw_parallax_image(window, parallax_pygame_layers, target_pos, center_of_moveable_space, args, SCALE, BORDERS)
            elif file_type == 'video':
                parallax_pygame_layers = frames[video_frame_iterator]
                draw_parallax_image(window, parallax_pygame_layers, target_pos, center_of_moveable_space, args, SCALE, BORDERS)
                video_frame_iterator = (video_frame_iterator + 1) % frame_count

            
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
