

import cv2
import os

from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError
from collections import defaultdict
import base64

from dotenv import load_dotenv
load_dotenv()

storage_name = os.getenv("STORAGE_ACCOUNT_NAME")
storage_url = os.getenv("STORAGE_ACCOUNT_URL")
container_name = os.getenv("CONTAINER_NAME")
container_sas_token = os.getenv("CONTAINER_SAS_TOKEN")
input_dataset_location = os.getenv("INPUT_DATASET_LOCATION")   
 
# Connect to the Blob Service using SAS token
blob_service_client = BlobServiceClient(account_url=storage_url, credential=container_sas_token)
# Reference the container
container_client = blob_service_client.get_container_client(container_name)

#how many frames per second to extract from the video
FRAMES_PER_SECOND = 4


#TODO - add recusive search for files in subfolders
#TODO - handle very large lists
def get_files_from_blob_storage(folder, ext):
    return [blob.name for blob in container_client.list_blobs(name_starts_with=folder) if blob.name.endswith(ext)]
    

def get_files_from_directory(root_path, extension=".mp4"):
    
    file_list = []
    # os.walk() generates the file names in a directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(extension):
                # Creates a full path to the file
                full_path = os.path.join(dirpath, file)
                file_list.append(full_path)
    return file_list

def read_file_content_from_blob_storage(file_path):
    try:
        blob_content = container_client.get_blob_client(file_path).download_blob()
        return blob_content.readall()
    except ResourceNotFoundError:
        print(f"Blob {file_path} not found.")
        return None
    
def video2frames(root_path):
    
    if not os.path.exists(root_path):
        raise Exception(f"Invalid input folder: {root_path}")
        
    mp4s = get_files_from_directory(root_path, ".mp4")

    for video in mp4s:
        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video)

        # Check if video opened successfully
        if not cap.isOpened():
            raise Exception(f"Error: Couldn't open the video file: {video}")

        # Get the video's frames per second rate -2
        fps = int(cap.get(cv2.CAP_PROP_FPS))


        # Define the desired output rate
        output_rate = FRAMES_PER_SECOND  
        output_folder = os.path.join(os.path.dirname(video), "frames", os.path.basename(video))
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Calculate the frame skip (how many frames to skip to get the desired rate)
        frame_skip = fps // output_rate

        frame_count = 0
        output_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                output_frame_name = os.path.join(output_folder, f"{output_frame_count}.jpg")
                if not cv2.imwrite(output_frame_name, frame):
                    raise Exception(f"Could not write image. Error in {output_folder}")
                output_frame_count += 1

            frame_count += 1

        # Release the video capture object
    cap.release()

    print(f"Saved {output_frame_count} frames at {output_rate} frames per second.")

#run video2frames. Uncomment if you want to run this script directly
#video2frames(input_dataset_location)
#get_files_from_blob_storage("zignalv2/2023/10/20/images/", ".jpg")
