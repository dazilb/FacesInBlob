# Face Cropping Tool for Identification Datasets

## Overview

This Python program is designed to streamline the process of cropping faces in images within an organized folder structure. It uses the RetinaFace face detection algorithm to identify and crop faces in images, saving each face as a separate image.

## How It Works

1. The program utilizes the RetinaFace face detection algorithm to identify faces in the input images.

2. If an image contains multiple faces, the program will crop each face individually.

3. The cropped face images are saved, preserving the original file's name and folder structure.

## Folder Structure Assumption

The program assumes the following folder structure:
```
Root
|-- Subdir1
| |-- File1.png
| |-- File2.png
|-- Subdir2
| |-- File1.png
|-- Subdir3
| |-- File1.png
```
The code should be placed at the root level of your dataset directory, with each subdirectory containing the image files you want to process. The program will automatically detect and crop faces in all images, saving the cropped face images in their respective subdirectories.

## Usage

1. Ensure you have the necessary dependencies installed. You may need to install RetinaFace or any other required libraries.

2. Place the code at the root level of your dataset directory.

3. Run the Python script.

4. The program will process each subdirectory, detect and crop faces in the images, and save the cropped face images in the respective subdirectories.

5. Check the subdirectories for the cropped face images.

## Installation

Before using this tool, you need to install the necessary Python libraries and configure your environment. Follow these steps to set up the required dependencies:

### 1. Install Python Dependencies

Make sure you have Python 3.9 or later installed. You can install the required libraries using `pip`:

```bash
pip install opencv-python numpy pillow azure-cognitiveservices-vision-face azure-storage-blob retinaface
```

### 2. Configure Azure Storage (if applicable)
If you plan to use Azure Blob Storage for image storage, make sure you have an Azure account and obtain the necessary Azure Blob Storage connection string. Replace "your_azure_blob_storage_connection_string" in the provided configuration file with your actual connection string.

## Configuration
To use the tool, you need to fill in the provided configuration file (config.json) with the relevant information. Here's how to configure the file:
```json lines
{
    "local_folder_path": "path_to_local_file",
    "connection_string": "your_azure_blob_storage_connection_string",
    "container_name_origin": "test",
    "container_name_result": "res",
    "is_local": true,
    "instance_key": "your_key",
    "endpoint": "your_endpoint"
}
```
