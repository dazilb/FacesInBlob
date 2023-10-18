import pathlib
import cv2
import os
import numpy as np
import json
import time

from PIL import Image

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.storage.blob import BlobServiceClient

# Retinaface
from retinaface import RetinaFace

# Init config
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

local_folder_path = config.get('local_folder_path')
connection_string = config.get('connection_string')
container_name_origin = config.get('container_name_origin')
container_name_result = config.get('container_name_result')
is_local = config.get('is_local')
instance_key = config.get('instance_key')
endpoint = config.get('endpoint')
count = 0


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

# -------------- Folder Processing -------------------
def process_images_in_azure_blob():
    blob_service_client, container_client_origin, container_client_results = init_blob_connection()
    for blob in container_client_origin.list_blobs():
        img_list = []
        image_name = blob.name
        local_image_name = os.getcwd() + pathlib.Path(image_name).name
        with open(file=local_image_name, mode='wb') as image_file:
            image_file.write(container_client_origin.download_blob(blob.name).readall())
        img_list.append(local_image_name)
        output_files = process_image_and_save_result(local_image_name)
        if not output_files:
            continue
        for idx, file_name in enumerate(output_files):
            face_image_name = os.path.splitext(image_name)[0] + "_" + str(idx) + os.path.splitext(image_name)[1]
            with open(file_name, 'rb') as face_image_file:
                container_client_results.upload_blob(name=face_image_name, data=face_image_file, overwrite=True)
            os.remove(file_name)
        os.remove(local_image_name)


def process_local_dir(root_folder):
    items = os.listdir(root_folder)
    for item in items:
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            process_images_in_folder(item_path)


def process_images_in_folder(directory_path):
    image_files = os.listdir(directory_path)
    for idx, file_name in enumerate(image_files):
        image_path = os.path.join(directory_path, file_name)
        process_image_and_save_result(image_path)


def process_image_and_save_result(image_path):
    cropped_images_filenames = []
    try:
        cropped_images = detect_faces_retinaface(image_path)
        if cropped_images is None:
            # don't need to crop images with one face
            return
        for idx, cropped_image in enumerate(cropped_images):
            new_file_name = os.path.splitext(image_path)[0] + "_" + str(idx) + os.path.splitext(image_path)[1]
            image = Image.fromarray(cropped_image)
            image.save(new_file_name)
            cropped_images_filenames.append(new_file_name)
        return cropped_images_filenames
    except Exception as e:
        print(f"Skipped {image_path} - {str(e)}")


def detect_faces_retinaface(image_path):
    starttime = time.time()
    cropped_images = []
    try:
        faces = extract_faces(image_path)
        print(time.time() - starttime)
        if faces is None or len(faces) < 2:
            return
        return faces
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def extract_faces(image_path):
    resp = []
    obj = RetinaFace.detect_faces(open_image(image_path))
    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]
            img = open_image(image_path)
            # The minimum detectable face size is 36 x 36 pixels https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-face-detection
            if facial_area[2] - facial_area[0] > 36 and facial_area[3] - facial_area[1] > 36:
                facial_img = crop_img(img, facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1])
                resp.append(facial_img[:, :, ::-1])
        return resp


def crop_img(img, x, y, width, height):
    h, w, _ = img.shape

    # Calculate the increase factor (50%)
    increase_factor = 0.15  # 50% increase

    # Calculate the new width and height
    new_width = int(width * (1 + increase_factor))
    new_height = int(height * (1 + increase_factor))

    # Calculate the new top-left corner (x, y) so that the center of the bounding box remains the same
    new_x = x - (new_width - width) // 2
    new_y = y - (new_height - height) // 2

    new_x = max(new_x, 0)
    new_y = max(new_y, 0)
    new_x = min(new_x, w - new_width)
    new_y = min(new_y, h - new_height)

    # Create the new bounding box
    return img[new_y:new_y + new_height, new_x:new_x + new_width]

    return img[facial_area1: facial_area3, facial_area1: facial_area2]


def detect_faces_azure(image_path):
    face_client = FaceClient(endpoint, CognitiveServicesCredentials(instance_key))
    cropped_images = []
    try:
            with open(image_path, "rb") as image_file:
                detected_faces = face_client.face.detect_with_stream(image=image_file)
            if len(detected_faces) < 2:
                return
            # Process and print information about detected faces
            for face in detected_faces:
                image = open_image(image_path)
                cropped_images.append(image[face.face_rectangle.top - face.face_rectangle.height : face.face_rectangle.height,
                                face.face_rectangle.left: face.face_rectangle.left + face.face_rectangle.width])
            return cropped_images
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def open_image(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    # Decode the image using OpenCV's imdecode function
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print("image is none")
        return
    return img


def init_blob_connection():
    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # Get a reference to the origin container
    container_client_origin = blob_service_client.get_container_client(container_name_origin)
    # Get a reference to the results container
    container_client_results = blob_service_client.get_container_client(container_name_result)
    return blob_service_client, container_client_origin, container_client_results


def resize_img(img):
    scale_percent = 400  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if is_local:
    process_local_dir(local_folder_path)
else:
    process_images_in_azure_blob()

