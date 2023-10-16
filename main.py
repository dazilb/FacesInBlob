import pathlib
import cv2
import os
import mediapipe as mp
import numpy as np
import imutils
import dlib
from azure.storage.blob import BlobServiceClient
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import json

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
dlib_face_detection = dlib.get_frontal_face_detector()

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
    for blob in container_client_origin.list_blobs():
        img_list = []
        image_name = blob.name
        local_image_name = "C:/Users/talshoham/Pictures/Test/" + pathlib.Path(image_name).name
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


# process_image
# get image path
# process image
# upload all the cropped images to the source folder
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

def crop_img(img, xmax, ymax, xmin, ymin):
    image_rows, image_cols, _ = img.shape
    image_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xmin -= 0.4 * (xmax - xmin)
    xmax += 0.4 * (xmax - xmin)
    ymin -= 0.4 * (ymax - ymin)
    ymax += 0.4 * (ymax - ymin)

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
    for idx, file_name in enumerate(IMAGE_FILES):
        image_path = os.path.join(directory_path, file_name)
        process_image_and_save_result(image_path)

def process_image_and_save_result(image_path):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        # Decode the image using OpenCV's imdecode function
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("image is none")
            return
        cropped_images = process_image_dlib(img)
        if cropped_images is None:
            # don't need to crop images with one face
            return
        for idx, cropped_image in enumerate(cropped_images):
            new_file_name = os.path.splitext(image_path)[0] + "_" + str(idx) + os.path.splitext(image_path)[1]
            retval, img_bytes = cv2.imencode(".jpg", cropped_image)
            with open(new_file_name, "wb") as f:
                f.write(img_bytes)
    except Exception as e:
        print(f"Skipped {image_path} - {str(e)}")


def process_image_mediapipe(image):
    cropped_images = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections is None or len(results.detections) < 2:
            # don't need to crop images with one face
            return
        for idx2, detection in enumerate(results.detections):
            location = detection.location_data

            relative_bounding_box = location.relative_bounding_box
            xmax = relative_bounding_box.xmin + relative_bounding_box.width
            ymax = relative_bounding_box.ymin + relative_bounding_box.height
            xmin = relative_bounding_box.xmin
            ymin = relative_bounding_box.ymin

            cropped_image = crop_img(image.copy(), xmax, ymax, xmin, ymin)
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            cropped_images.append(cropped_image)
    return cropped_images

def process_image_dlib(image):
    cropped_images = []
    #small_image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = dlib_face_detection(rgb, 1)
    if results is None or len(results) < 2:
        # don't need to crop images with one face
        return
    for idx, detection in enumerate(results):
        startX, startY, w, h = convert_and_trim_bb(image, detection)
        cropped_image = image[startY: startY + h, startX: startX + w]
        cropped_images.append(cropped_image)
    return cropped_images

def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)

# def process_image(file):
#     detected_faces_list = []
#     with mp_face_detection.FaceDetection(
#             model_selection=1, min_detection_confidence=0.3) as face_detection:
#             try:
#                 image = cv2.imread(file)
#                 # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
#                 results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#                 if results.detections is None:
#                     # don't need to crop images with one face
#                     # @daria - maybe we want to log this condition? so that we can validate this image?
#                     return
#                 for idx, detection in enumerate(results.detections):
#                     cropped_image = image.copy()
#                     cropped_image = crop_img(cropped_image,detection)
#                     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
#                     detected_faces_list.append(cropped_image)
#                 return detected_faces_list
#             except Exception as e:
#                 print(f"Skipped {file} - {str(e)}")

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

