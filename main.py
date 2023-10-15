import time
from PIL import Image
import cv2
import os
import mediapipe as mp
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

# get Azure Storage connection string and container name from the config file
connection_string = "your_connection_string_here"
container_name = "your_container_name_here"

# Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get a reference to the container
container_client = blob_service_client.get_container_client(container_name)

# Function to convert an image to grayscale
def convert_to_grayscale(image_path):
    image = Image.open(image_path)
    grayscale_image = image.convert("L")
    return grayscale_image

# Iterate through all blobs in the container
def crop_all_images_in_blob():
    for blob in container_client.list_blobs():
        # Check if the blob is a directory (folder)
        img_list = []
        if blob.name.endswith('/'):
            folder_name = blob.name.rstrip('/')
            for image_blob in container_client.walk_blobs(folder_name):
                image_name = image_blob.name
                image_data = container_client.get_blob_client(image_name).download_blob().readall()
                with open(image_name, 'wb') as image_file:
                    image_file.write(image_data)
                img_list.append(image_name)
                faces = process_image(image_name)
                if not faces:
                    continue
                for idx, face in faces:
                    face_image_name = os.path.join(folder_name, f'cropped_{folder_name}_{idx}')
                    face.save(face_image_name)
                    with open(face_image_name, 'rb') as grayscale_image_file:
                        container_client.upload_blob(grayscale_image_file, name=face_image_name)
                    os.remove(image_name)
                os.remove(face_image_name)

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

    xmin = xmin if xmin > 0 else 0
    ymin = ymin if ymin > 0 else 0
    xmax = xmax if xmax < image_cols else image_cols
    ymax = ymax if ymax < image_rows else image_rows

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

def process_images():
    folder_path = "C:/test"
    IMAGE_FILES = os.listdir(folder_path)
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.3) as face_detection:
        for idx, file_name in enumerate(IMAGE_FILES):
            file = os.path.join(folder_path, file_name)
            try:
                image = cv2.imread(file)
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.detections < 2:
                    # don't need to crop images with one face
                    continue
                for idx2, detection in enumerate(results.detections):
                    cropped_image = image.copy()
                    cropped_image = crop_img(cropped_image,detection)
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(file + str(idx2) + '.png', cropped_image)
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
                if results.detections < 2:
                    # don't need to crop images with one face
                    return
                for idx, detection in enumerate(results.detections):
                    cropped_image = image.copy()
                    cropped_image = crop_img(cropped_image,detection)
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                    detected_faces_list.append(cropped_image)
                return detected_faces_list
            except Exception as e:
                print(f"Skipped {file} - {str(e)}")


