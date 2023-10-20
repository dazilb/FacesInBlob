
import os

from florence_face_detection import detect_face, crop_faces
from misc import get_files_from_directory, get_files_from_blob_storage, read_file_content_from_blob_storage

from dotenv import load_dotenv
load_dotenv()

#input_dataset_location = os.getenv("INPUT_DATASET_LOCATION")   
#images = get_files_in_directory( os.path.join(input_dataset_location,"frames"), ".jpg")

input_dataset_location = os.getenv("INPUT_CONTAINER_FOLDER_NAME")
output_location = os.getenv("OUTPUT_LOCATION")

print (f"input_dataset_location={input_dataset_location}")

images = get_files_from_blob_storage(input_dataset_location, ".jpg")

for path_to_image in images:
    image_content = read_file_content_from_blob_storage(path_to_image)
    faces = detect_face(image_content)
    if faces:
        crop_faces(faces, image_content, os.path.dirname(output_location) + "/cropped/", os.path.splitext(path_to_image.replace("/", "_"))[0]) #, os.path.basename(path_to_image)
    else:
        print(f"No faces detected for image: {path_to_image}")
        