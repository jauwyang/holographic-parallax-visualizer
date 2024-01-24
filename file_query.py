import json


METADATA_FILE = 'input_metadata.json'
EMPTY_FILE_FORMAT = {
    'files': []
}

def input_exists(input_filename):
    try:
        with open(METADATA_FILE, 'r') as file:
            json_data = json.load(file)
    except:
        create_metadata_file()
        with open(METADATA_FILE, 'r') as file:
            json_data = json.load(file)
    
    for metadata in json_data['files']:
        if metadata['input_src_name'] == input_filename:
            return True
    return False


def create_metadata_file():
    with open(METADATA_FILE, 'w') as file:
        json.dump(EMPTY_FILE_FORMAT, file, indent=2)


def read_metadata(input_filename):
    with open(METADATA_FILE, 'r') as file:
        json_data = json.load(file)
    
    for metadata in json_data['files']:
        if metadata['input_src_name'] == input_filename:
            return metadata
        
    print('ERROR: Desired file metadata not found when it should exist.')


def write_metadata(input_filename, frames, frame_count):
    data = {
        'input_src_name': input_filename,
        'frame_count': frame_count,
        'frames': frames,
    }
    
    with open(METADATA_FILE, 'r+') as file:
        # Get current data
        json_data = json.load(file)
        json_data['files'].append(data)
        
        file.seek(0)  # reset read/write position to start
        
        json.dump(json_data, file, indent=2)
