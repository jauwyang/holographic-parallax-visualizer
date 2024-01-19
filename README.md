# Holographic Parallax Visualizer

## Summary

This program converts any standard 2D image (e.g. .jpg, .png, etc.) into a 3D holographic image with a parallax effect. It tracks the position of the user's face and shifts the parallax perspective accordingly to create a 3D depth illusion. The program can alternatively track the position of the mouse instead.

## How It Works

![](/markdown_sample_assets/gwen_stacy_parallax.gif)

TLDR: The program computes the relative depth of all elements of an image and groups regions of similar depths into different layers that shift at varying speeds based on their depth - closer layers move faster, further layers move slower.

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

## How to Use

Create a virtual environment: `python -m venv venv`

Enter the virtual environment: `venv\Scripts\activate`

Install required python packages: `pip install -r requirements.txt`

Run the program: 

`python app.py --model 'large' --view '2d' --controller 'mouse'`





MiDaS (PyTorch) uses CUDA 12.1 (NViDiA)