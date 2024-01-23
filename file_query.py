import cv2
import os
import json

IMAGE_METADATA_FILE = 'input_images_metadata.json'
VIDEO_METADATA_FILE = 'input_videos_metadata.json'

EMPTY_IMAGE_FILE_FORMAT = {
    'images': []
}
EMPTY_VIDEO_FILE_FORMAT = {
    'videos': []
}


# --- Images ---

def input_image_exists(filename):
    try:
        with open(IMAGE_METADATA_FILE, 'r') as file:
            json_data = json.load(file)
    except:  # if file content doesnt exist, create it and open again
        create_image_metadata_file()
        with open(IMAGE_METADATA_FILE, 'r') as file:
            json_data = json.load(file)
    
    for source in json_data['images']:
        if source['input_src_name'] == filename:
            return True
    
    return False


def create_image_metadata_file():
    with open(IMAGE_METADATA_FILE, 'w') as file:
        json.dump(EMPTY_IMAGE_FILE_FORMAT, file, indent=2)


def read_image_metadata(input_name):
    with open(IMAGE_METADATA_FILE, 'r') as file:
        file_data = json.load(file)
    
    for metadata in file_data['images']:
        if metadata['input_src_name'] == input_name:
            return metadata
    
    print('ERROR: Image metadata not found when it should have.')


def write_image_metadata(input_name, layers):
    data = {
        'input_src_name': input_name,
        'parallax_layers': layers,
    }

    with open(IMAGE_METADATA_FILE, 'r+') as file:
        # Get current data
        file_data = json.load(file)
        file_data['images'].append(data)
        
        file.seek(0)  # reset read/write position to start
        
        json.dump(file_data, file, indent=2)


# --- Videos ---

def input_video_exists(filename):
    try:
        with open(VIDEO_METADATA_FILE, 'r') as file:
            json_data = json.load(file)
    except:  # if file content doesnt exist, create it and open again
        create_video_metadata_file()
        with open(VIDEO_METADATA_FILE, 'r') as file:
            json_data = json.load(file)
    
    for source in json_data['videos']:
        if source['input_src_name'] == filename:
            return True
    
    return False

def create_video_metadata_file():
    with open(VIDEO_METADATA_FILE, 'w') as file:
        json.dump(EMPTY_VIDEO_FILE_FORMAT, file, indent=2)


def read_video_metadata(input_name):
    with open(VIDEO_METADATA_FILE, 'r') as file:
        file_data = json.load(file)
        
    for metadata in file_data['videos']:
        if metadata['input_src_name'] == input_name:
            return metadata
    
    print('ERROR: Video metadata not found when it should have.')


def write_video_metadata(input_name, frames, frame_count):
    data = {
        'input_src_name': input_name,
        'frame_count': frame_count,
        'frames': frames,
    }
    
    with open(VIDEO_METADATA_FILE, 'r+') as file:
        # Get current data
        file_data = json.load(file)
        file_data['videos'].append(data)
        
        file.seek(0)  # reset read/write position to start
        
        json.dump(file_data, file, indent=2)

    

def delete_all_data():
    print("Deleting all image/video metadata...")
    print("Note: dir containing images must be manually deleted by going to './output/[images or videos]/[target dir]'")
    with open('input_images_metadata.json', 'w') as file:
        print("Deleted Image Metadata")
    
    with open('input_videos_metadata.json', 'w') as file:
        print("Deleted Video Metadata")
