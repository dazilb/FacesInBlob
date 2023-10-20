

'''returnFaceId: false
returnFaceLandmarks: true
returnFaceAttributes: mask
recognitionModel: recognition_01
returnRecognitionModel: false
detectionModel: detection_03
faceIdTimeToLive: 86400 '''

#/face/v1.0/detect?
#https://cognitiveservice-303474.cognitiveservices.azure.com/face/v1.0/detect?returnFaceId=false&returnFaceLandmarks=true&returnFaceAttributes=mask&recognitionModel=recognition_01&returnRecognitionModel=false&detectionModel=detection_03&faceIdTimeToLive=86400

import os
import requests
import json
import cv2
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from misc import read_file_content_from_blob_storage

# Your Azure endpoint and subscription key
cognitive_service_endpoint = os.getenv("COGNITIVE_SERVICES_ENDPOINT")        #"https://cognitiveservice-303474.cognitiveservices.azure.com/"
cognitive_service_subscription_key = os.getenv("COGNITIVE_SERVICES_KEY") 

# API URL
face_api_url = cognitive_service_endpoint + "face/v1.0/detect"

def detect_face(image_content):

    # Set image_url to the URL of an image that you want to analyze.
    # Read the local image file
    #with open(path_to_image, 'rb') as image_file:
    #    image_data = image_file.read()
    #image_content = read_file_content_from_blob_storage(path_to_image)

    headers = {
        'Ocp-Apim-Subscription-Key': cognitive_service_subscription_key,
        'Content-Type': 'application/octet-stream'
    }

    params = {
        'returnFaceId': 'false',
        'returnFaceLandmarks': 'false',
        'recognitionModel': 'recognition_04',
        'returnRecognitionModel': 'false',
        'detectionModel': 'detection_03'
    }
    start = time.time()
    response = requests.post(face_api_url, headers=headers, params=params, data=image_content)
    print(f"Time to detect face = {time.time() - start}")
    faces = response.json()
    return faces


def crop_faces(faces, image_content, output_dir_path, image_name, padding_percentage=0.3):
    
    if image_content is not None:
        np_array = np.frombuffer(image_content, dtype=np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    else:
        raise Exception(f"Could not read image. Error in {image_name}")
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    img_height, img_width, _ = img.shape
    
    face_no = 0
    for face in faces:
        rectangle = face['faceRectangle']
        x, y, w, h = rectangle['left'], rectangle['top'], rectangle['width'], rectangle['height']
        
         # Calculate padding
        x_padding = int(w * padding_percentage)
        y_padding = int(h * padding_percentage)
        
         # Adjust the coordinates with padding
        x_start = max(x - x_padding, 0)
        y_start = max(y - y_padding, 0)
        x_end = min(x + w + x_padding, img_width)
        y_end = min(y + h + y_padding, img_height)
        
        # Crop the face from the image
        #cropped_face = img[y:y+h, x:x+w]
        cropped_face = img[y_start:y_end, x_start:x_end]
        
        if not cv2.imwrite( os.path.join(output_dir_path, str(face_no) + "_" + image_name + ".jpg" ), cropped_face):
            raise Exception(f"Could not write image. Error in {image_name}")
        face_no+=1

def get_all_files_in_directory(script_path):
    
    file_list = []
    
    # os.walk() generates the file names in a directory tree
    for dirpath, dirnames, filenames in os.walk(script_path):
        for file in filenames:
            # Creates a full path to the file
            full_path = os.path.join(dirpath, file)
            file_list.append(full_path)
    return file_list



    