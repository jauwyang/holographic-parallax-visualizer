import torch
import cv2
import os

# MIDAS
# https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-01-08-depth-vision-midas/2023-01-08/#choosing-the-right-model

MIDAS_SMALL = "MiDaS_small"  # (highest accuracy, slowest inference speed)
MIDAS_MEDIUM = "DPT_Hybrid"  # (medium accuracy, medium inference speed)
MIDAS_LARGE = "DPT_Large"  # (lowest accuracy, highest inference speed)

# Load Model
def setup_prediction_settings(input_params):
    if input_params == 'small':
        selected_model = MIDAS_SMALL
    elif input_params == 'medium':
        selected_model = MIDAS_MEDIUM
    elif input_params == 'large':
        selected_model = MIDAS_LARGE
    else:
        print("Invalid Model")
        return
    
    midas = torch.hub.load("intel-isl/MiDaS", str(selected_model))
    
    # Choose Device (to run prediction on: CUDA/Nvidia vs. CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    midas.to(device)
    midas.eval()

    # Choose Transformation Pipeline
    #   - inputs need to be 'transformed' to match the dataset format the model was trained with.
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if (selected_model == MIDAS_LARGE or selected_model == MIDAS_MEDIUM):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
        
    return device, midas, transform 
        
        
def create_depth_map(selected_model):
    device, midas_model, transform = setup_prediction_settings(selected_model)
    
    input_image_dir_path = './input_images'
    image_filename = os.listdir(input_image_dir_path)[0]
    input_image = cv2.imread(os.path.join(input_image_dir_path, image_filename))

    image_midas = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    imgbatch = transform(image_midas).to(device)

    with torch.no_grad():
        prediction = midas_model(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    output = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    cv2.imwrite('./rgbd_images/colour_img.jpg', input_image)
    cv2.imwrite('./rgbd_images/depth_img.jpg', output)


def draw_depth_map():
    depth_map = cv2.imread('./rgbd_images/depth_img.jpg')
    
    output_cv2_display = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Depth Map', output_cv2_display)