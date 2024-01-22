import cv2
import os
import json


def get_file_params(input_type):
    """
    Returns specific params needed for 
    """
    if input_type == 'image':
        metadata_file = 'input_images_metadata.json'
        empty_json_init = {
            'images': []
        }
        
    elif input_type == 'video':
        metadata_file = 'input_videos_metadata.json'
        empty_json_init = {
            'videos': []
        }
    
    return metadata_file, empty_json_init



def input_exists(filename, input_type):
    """
    Checks if source input has already been computed & stored
    """
    metadata_file, empty_json_init = get_file_params(input_type)
    
    try:
        with open(metadata_file, 'r') as file:
            json_data = json.load(file)
    except:  # if file content doesnt exist, create it and open again
        initialize_input_metadata(input_type)
        with open(metadata_file, 'r') as file:
            json_data = json.load(file)
    
    for source in json_data[str(input_type) + 's']:
        if source['input_src_name'] == filename:
            return True
    
    return False



def initialize_input_metadata(input_type):
    """
    Initializes json file format
    """
    metadata_file, empty_json_init = get_file_params(input_type)
    
    with open(metadata_file, 'w') as file:
        json.dump(empty_json_init, file, indent=2)



def write_image_metadata(input_name, layers):
    data = {
        'input_src_name': input_name,
        'parallax_layers': layers,
    }
    
    metadata_file, empty_json_init = get_file_params('image')
    
    with open(metadata_file, 'r+') as file:
        # Get current data
        file_data = json.load(file)
        file_data['images'].append(data)
        
        file.seek(0)  # reset read/write position to start
        
        json.dump(file_data, file, indent=2)


def read_image_metadata(input_name):
    metadata_file, empty_json_init = get_file_params('image')
    with open(metadata_file, 'r') as file:
        file_data = json.load(file)
    
    for metadata in file_data['images']:
        if metadata['input_src_name'] == input_name:
            return metadata
    
    print('File not found: ERROR')
    

def delete_all_data():
    print("Deleting all image/video metadata...")
    print("Note: dir containing images must be manually deleted by going to './output/[images or videos]/[target dir]'")
    with open('input_images_metadata.json', 'w') as file:
        print("Deleted Image Metadata")
    
    with open('input_videos_metadata.json', 'w') as file:
        print("Deleted Video Metadata")
    
    
# def video2frames(video_path, output_path):
#     cap = cv2.VideoCapture(video_path)
    
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
        
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
    
#     for frame_index in range(frame_count):
#         ret, frame = cap.read()
        
#         if not ret:
#             break
        
#         frame_path = os.path.join(output_path)
        
#     frames = []
    
#     cap.release()
    
#     return frames
