

import cv2
import os 

FRAMES_PER_SECOND = 4

def get_files_in_directory(root_path, extension=".mp4"):
    
    file_list = []
    
    # os.walk() generates the file names in a directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(extension):
                # Creates a full path to the file
                full_path = os.path.join(dirpath, file)
                file_list.append(full_path)
    return file_list


def video2frames(root_path):
    
    if not os.path.exists(root_path):
        raise Exception(f"Invalid input folder: {root_path}")
        
    # Path to the video file
    #video_path = '/Users/vladfeigin/myprojects/facerecognition/data/face_detection_dataset/red_556_orionprodrawdata-zgn-zignalv220231017videosRGr8cSeJ6T2ct0Rzmp4.mp4'

    mp4s = get_files_in_directory(root_path, ".mp4")

    for video in mp4s:
        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video)

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Couldn't open the video file.")
            exit()

        # Get the video's frames per second rate -2
        fps = int(cap.get(cv2.CAP_PROP_FPS))


        # Define the desired output rate
        output_rate = FRAMES_PER_SECOND  
        output_folder =  os.path.dirname(video) + "/frames/" + os.path.basename(video) 
        
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
        
                output_frame_name = output_folder + "/" +str(output_frame_count) + ".jpg"
                if not cv2.imwrite(output_frame_name, frame):
                    raise Exception(f"Could not write image. Error in {output_folder}")
                output_frame_count += 1

            frame_count += 1

        # Release the video capture object
    cap.release()

    print(f"Saved {output_frame_count} frames at {output_rate} frames per second.")


video2frames("/Users/vladfeigin/myprojects/facerecognition/data/face_detection_dataset")
