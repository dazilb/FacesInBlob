import pathlib
import cv2
import os
import mediapipe as mp
import numpy as np
import PIL
from PIL import Image
from azure.storage.blob import BlobServiceClient
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# The code in this file assumes that the root folder contains folders of images, 
# In the following directory structure (for both the cloud and local implementations):
# Root
# |-- Subdir1
# |   |-- File1.png
# |   |-- File2.png
# |-- Subdir2
# |   |-- File1.png
# |-- Subdir3
# |   |-- File1.png
local_folder_path = config.get('local_folder_path')
connection_string = config.get('connection_string')
container_name_origin = config.get('container_name_origin')
container_name_result = config.get('container_name_result')
is_local = config.get('is_local')


# Iterate through all blobs in the container
def crop_all_images_in_container():
    blob_service_client, container_client_origin, container_client_results = init_blob_connection()
    for blob in container_client_origin.list_blobs(): # This goes through all blobs in container in all the folders
        img_list = []
        image_name = blob.name
        local_image_name="C:/Users/talshoham/Pictures/Test/" + pathlib.Path(image_name).name
        with open(file=local_image_name, mode='wb') as image_file:
            image_file.write(container_client_origin.download_blob(blob.name).readall())
        img_list.append(local_image_name)
        faces = process_image(local_image_name)
        if not faces:
            continue
        for idx, face in enumerate(faces):
            face_image_name = os.path.splitext(image_name)[0] + "_" + str(idx) + os.path.splitext(image_name)[1]
            local_face_name = os.path.splitext(local_image_name)[0] + "_" + str(idx) + os.path.splitext(local_image_name)[1]
            cv2.imwrite(local_face_name, face)
            with open(local_face_name, 'rb') as face_image_file:
                container_client_results.upload_blob(name=face_image_name, data=face_image_file, overwrite=True)
            os.remove(local_face_name)
        os.remove(local_image_name)

def process_images_from_folder(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
    return image_list

def resize_img(img):
    scale_percent = 400  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def crop_img(img, detection):
    image_rows, image_cols, _ = img.shape
    image_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    location = detection.location_data

    relative_bounding_box = location.relative_bounding_box
    xmax = relative_bounding_box.xmin + relative_bounding_box.width
    ymax = relative_bounding_box.ymin + relative_bounding_box.height
    xmin = relative_bounding_box.xmin
    ymin = relative_bounding_box.ymin

    xmin -= 0.4 * (xmax - relative_bounding_box.xmin)
    xmax += 0.4 * (xmax - relative_bounding_box.xmin)
    ymin -= 0.4 * (ymax - relative_bounding_box.ymin)
    ymax += 0.4 * (ymax - relative_bounding_box.ymin)

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, 1)
    ymax = min(ymax, 1)

    rect_start_point = _normalized_to_pixel_coordinates(
        xmin, ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        xmax,
        ymax, image_cols,
        image_rows)

    xleft, ytop = rect_start_point
    xright, ybot = rect_end_point

    return image_input[ytop: ybot, xleft: xright]

def process_dir(root_folder):
    items = os.listdir(root_folder)
    for item in items:
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path): 
            process_images_in_folder(item_path)

def process_images_in_folder(directory_path):
    IMAGE_FILES = os.listdir(directory_path)
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.3) as face_detection:
        for idx, file_name in enumerate(IMAGE_FILES):
            file = os.path.join(directory_path, file_name)
            try:
                image = Image.open(file)
                cv2_image = np.array(image)
                results = face_detection.process(cv2_image)
                if results.detections is None or len(results.detections) < 2:
                    # don't need to crop images with one face
                    continue
                for idx2, detection in enumerate(results.detections):
                    cropped_image = cv2_image.copy()
                    cropped_image = crop_img(cropped_image, detection)
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                    img_pil = Image.fromarray(cropped_image)
                    new_file_name = os.path.splitext(file)[0] + "_" + str(idx2) + os.path.splitext(file)[1] or ".jpg"
                    img_pil.save(new_file_name)
            except Exception as e:
                print(f"Skipped {file} - {str(e)}")

def process_image(file):
    detected_faces_list = []
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.3) as face_detection:
            try:
                image = cv2.imread(file)
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.detections is None:
                    # don't need to crop images with one face
                    # @daria - maybe we want to log this condition? so that we can validate this image?
                    return
                for idx, detection in enumerate(results.detections):
                    cropped_image = image.copy()
                    cropped_image = crop_img(cropped_image,detection)
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                    detected_faces_list.append(cropped_image)
                return detected_faces_list
            except Exception as e:
                print(f"Skipped {file} - {str(e)}")

def init_blob_connection():
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # Get a reference to the origin container
    container_client_origin = blob_service_client.get_container_client(container_name_origin)
    # Get a reference to the results container
    container_client_results = blob_service_client.get_container_client(container_name_result)
    return blob_service_client, container_client_origin, container_client_results

if(is_local):
    process_dir(local_folder_path)
else:
    crop_all_images_in_container()

