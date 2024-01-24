import argparse
import cv2
import os
import pygame as pg

from config import SCALE
from depth_map import create_depth_map
from layered_parallax import (
    get_parallax_layers,
    create_parallax_layers,
    draw_parallax_frame,
    cv2pygame_image,
)
from utils import Vector2D, print_video_progression
from camera import Camera
from file_query import (
    input_exists,
    read_metadata,
    write_metadata,
)
from video import video2frames

BORDERS = 20
INPUT_DIR = "input_src"

# output/
#         # | -- images/
#         # |    | -- image01/
#         # |    |    | -- frames/
#         # |    |    |    | -- frame001/
#         # |    |    |    |    | -- depth_map/
#         # |    |    |    |    |    | -- depth_map.png
#         # |    |    |    |    | -- layers/
#         # |    |    |    |    |    | -- layer001.png
#         # |    |    |    |    |    | -- layer002.png
#         # |    |    |    |    |    | -- layer003.png
#         # |
#         # | -- videos/
#         # |    | -- video01/
#         # |    |    | -- frames/
#         # |    |    |    | -- frame001/
#         # |    |    |    |    | -- depth_map/
#         # |    |    |    |    |    | -- depth_map.png
#         # |    |    |    |    | -- layers/
#         # |    |    |    |    |    | -- layer001.png
#         # |    |    |    |    |    | -- layer002.png
#         # |    |    |    |    |    | -- layer003.png
#         # |    |    |    | -- frame002/
#         # |    |    |    |    | -- depth_map/
#         # |    |    |    |    |    | -- depth_map.png
#         # |    |    |    |    | -- layers/
#         # |    |    |    |    |    | -- layer001.png
#         # |    |    |    |    |    | -- layer002.png
#         # |    |    |    |    |    | -- layer003.png


class App:
    def __init__(self):
        pg.init()
        self.run()

    def run(self):
        # Get user parameters
        args = self.parse_arguments()

        input_file, input_type = args.input, args.intype
        input_file_name = input_file.split(".")[0]  # name without extension

        file_output_path_root = os.path.join(
            "output", input_type + "s", input_file_name
        )
        file_output_path_all_frames = os.path.join(file_output_path_root, "frames")



        # Write/build data (if it doesn't exist)
        if input_exists(input_file):
            print("Input already computed.\nNow generating visualizer.")
        else:
            if os.path.exists(file_output_path_root):
                print(
                    "ERROR: Output directory for file already exists but has no metadata"
                )
                return

            os.makedirs(file_output_path_root)
            os.makedirs(file_output_path_all_frames)

            if input_type == "video":
                file_frames = video2frames(os.path.join(INPUT_DIR, input_file))
            elif input_type == "image":
                file_frames = [cv2.imread(os.path.join(INPUT_DIR, input_file))]

            frame_count = len(file_frames)

            frames = {}

            for i, current_frame in enumerate(file_frames):
                print_video_progression(i, frame_count)
                
                counter = str(i).zfill(4)

                file_output_path_current_frame = os.path.join(
                    file_output_path_all_frames, "frame" + counter
                )
                os.makedirs(file_output_path_current_frame)

                os.makedirs(os.path.join(file_output_path_current_frame, "depth_map"))
                os.makedirs(os.path.join(file_output_path_current_frame, "layers"))

                create_depth_map(
                    current_frame, input_type, file_output_path_current_frame
                )

                layer_names = create_parallax_layers(
                    current_frame, file_output_path_current_frame
                )

                frames["parallax_layers" + counter] = layer_names

            print_video_progression(frame_count, frame_count)
            
            write_metadata(input_file, frames, frame_count)
            print("Finished building frames & layers")

        if args.onlybuild:
            return



        # Read data
        src_data = read_metadata(input_file)

        frames = []
        frame_count = src_data["frame_count"]

        for i in range(frame_count):
            counter = str(i).zfill(4)

            file_output_path_current_frame = os.path.join(
                file_output_path_all_frames, "frame" + counter
            )

            parallax_frame_layers = get_parallax_layers(
                file_output_path_current_frame,
                src_data["frames"]["parallax_layers" + counter],
            )

            pygame_frame_layers = []
            width, height = (
                parallax_frame_layers[0].shape[1],
                parallax_frame_layers[0].shape[0],
            )

            for layer in parallax_frame_layers:
                pygame_frame_layers.append(
                    pg.transform.scale(
                        cv2pygame_image(layer, mode="RGBA"),
                        (int(width * SCALE), int(height * SCALE)),
                    )
                )

            frames.append(pygame_frame_layers)



        # Create Pygame window
        window = pg.display.set_mode(
            (int((width - BORDERS) * SCALE), int((height - BORDERS) * SCALE))
        )
        pg.display.set_caption("Parallax Image")

        # Setup controller
        if args.controller == "face":
            cam = Camera()
            center_of_moveable_space = Vector2D(
                cam.dimensions.width * SCALE / 2, cam.dimensions.height * SCALE / 2
            )
        elif args.controller == "mouse":
            cam = None
            center_of_moveable_space = Vector2D(width * SCALE / 2, height * SCALE / 2)



        # Main Loop
        running = True
        frame_iterator = 0
        while running:
            # Get position of point of interest/controller (eyes or mouse)
            if args.controller == "face":
                cam.update_frame()
                eyes = cam.track_eye()
                cam.draw_eye_tracking(eyes)
                x, y, w, h = eyes
                target_pos = (x + (w / 2), y + (h / 2))

            elif args.controller == "mouse":
                x, y = pg.mouse.get_pos()
                target_pos = Vector2D(x, y)

            # Display Parallax Image(s)
            parallax_pygame_layers = frames[frame_iterator]
            draw_parallax_frame(
                window,
                parallax_pygame_layers,
                target_pos,
                center_of_moveable_space,
                args,
                SCALE,
                BORDERS,
            )
            frame_iterator = (frame_iterator + 1) % frame_count

            # Exit Methods (close pygame window or 'q' on camera)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            if cv2.waitKey(1) == ord("q"):
                running = False

        self.quit(cam)


    def quit(self, cam):
        pg.quit()
        if cam is not None:
            del cam
        cv2.destroyAllWindows()


    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Specify process settings")
        parser.add_argument("--input", type=str, required=True)
        parser.add_argument(
            "--intype", type=str, choices=["image", "video"], required=True
        )
        parser.add_argument(
            "--controller", type=str, choices=["face", "mouse"], required=True
        )
        parser.add_argument("-onlybuild", action="store_true")

        return parser.parse_args()


if __name__ == "__main__":
    app = App()
