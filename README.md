# Holographic Parallax Visualizer

## Summary

This program converts any standard 2D image or video (e.g. .jpg, .mp4, etc.) into a 3D holographic scene with a parallax effect. It tracks the position of the user's face and shifts the parallax perspective accordingly to create a 3D depth illusion. The program can alternatively track the position of the mouse instead.

## How It Works

![](/markdown_sample_assets/gwen_stacy_parallax.gif)

For simplicity, let's explain using a single framed image.

The program takes the desired image as input (in `/input_images`) and passes it into the [MiDaS Model](https://pytorch.org/hub/intelisl_midas_v2/) which computes the relative depth from this single image and outputs the corresponding depth map image.

Below are sample input and depth map images (depth map is converted to greyscale below)
![Input sample image](/markdown_sample_assets/gwen_stacy_src.JPG "Sample Input")

![Input sample image](/markdown_sample_assets/gwen_stacy_depth_map.jpg "Sample Depth map computed by MiDaS in grayscale")


The depth map is then divided into different layers according to each pixel's greyscale pixel value where lighter colours are closer to the camera and darker colours are further away. For instance, if layers are separated by ranges of 30, then pixels with values between (0) to (30) will be grouped into one layer, (31 to 60) into another, and so on. Visual examples are shown below.

![Example parallax image layers separated](/markdown_sample_assets/separated_parallax_sample_layers.jpg)

![Example parallax image layers merged](/markdown_sample_assets/combined_parallax_sample_layers.jpg)

Since each layer is cropped from separate regions of the image,  shifting the layers (e.g. apart) will create gaps in the final image. To fix this, the layer behind a given layer must be "inpainted" to fill in those gaps. 

![Gaps between layers](/markdown_sample_assets/gaps_in_layers.JPG)
![Inpainting between layers](/markdown_sample_assets/inpainted_layers.JPG)

These layers will then be placed ontop of each other with the "furthest" layer as the base and the "closest" layer on top.

Each layer will be shifted relative to a "focal point" - this is either the tracked position of the user's face or mouse. The movement speeds of these layers are based on relative depth of each layer - closer layers move faster, further layers move slower.

***TLDR***: The program computes the relative depth of all elements of an image and groups regions of similar depths into different layers that shift at varying speeds based on their depth - closer layers move faster, further layers move slower.

For videos, the process above is repeated for every frame in the clip. This takes a much longer time to process the entire scene, so the MiDaS depth model used is lighter than when selecting an `image` as input. Regardless, the computing time for a 720p video took about 30 seconds for each frame or 15 minutes to process 1 second in a 30 fps video.

## Other Examples

### Videos

AVENGERS

![](/markdown_sample_assets/regular_avengers.gif)
![](/markdown_sample_assets/parallax_sensitive_avengers.gif)

WONDER WOMAN

![](/markdown_sample_assets/regular_wonderwoman.gif)
![](/markdown_sample_assets/parallax_sensitive_wonderwoman.gif)


**Recommendations when using:**: 
* increase the sensitivity of the parallax effect for videos since the movements of the scene make the holographic effect less noticeable.
* stay roughly below 10s in length for videos
* can decrease the scale of the output video (`SCALE`) to decrease memory usage during visualization
* make sure there are no black bars in the input
* try testing the video before computing it (avoid long wait time for bad result) by inputting 1 frame image of the shot. 


## How to Use

Create a virtual environment: `python -m venv venv`

Enter the virtual environment: `venv\Scripts\activate`

Install required python packages: `pip install -r requirements.txt`

Run the program: 

`python app.py --controller 'mouse' --input 'avengers.mp4' --intype 'video'`

`python app.py --controller 'mouse' --input 'crowded_room.JPG' --intype 'image'`





MiDaS (PyTorch) uses CUDA 12.1 (NViDiA)