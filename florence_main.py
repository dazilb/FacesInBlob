
import os

from florence_face_detection import detect_face, crop_faces
from misc import get_files_in_directory

from dotenv import load_dotenv
load_dotenv()

input_dataset_location = os.getenv("INPUT_DATASET_LOCATION")   
frames = get_files_in_directory( os.path.join(input_dataset_location,"frames"), ".jpg")

for path_to_image in frames:
    faces = detect_face(path_to_image)
    if faces:
        crop_faces(faces, path_to_image, os.path.dirname(path_to_image) + "/cropped/",  os.path.basename(path_to_image))
    else:
        print(f"No faces detected for image: {path_to_image}")
        