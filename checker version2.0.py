from PIL import Image
import os

def compare_images(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    if img1.size != img2.size:
        return False
    pixels1 = img1.load()
    pixels2 = img2.load()
    for i in range(img1.size[0]):
        for j in range(img1.size[1]):
            if pixels1[i,j] != pixels2[i,j]:
               # print("False")
                return False
    print("True")
    return True

# Define the path to the directory containing the 132 folders
parent_dir = r"D:\PDARS\datasetiso\army worm"

def processor(image1_path, parent_dir):
# Loop over each subdirectory in the parent directory
    for subdir in os.listdir(parent_dir):
        # Define the path to the current subdirectory
        subdir_path = os.path.join(parent_dir, subdir)
    
        # Check if the current subdirectory is a directory (and not a file)
        if os.path.isdir(subdir_path):
            # Loop over each file in the current subdirectory
            for filename in os.listdir(subdir_path):
                # Define the path to the current file
                file_path = os.path.join(subdir_path, filename)
            
                # Check if the current file is an image file
                if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    # Call the process_image function with the current file path
                    compare_images(image1_path, file_path)
                    print("okay")


image1_path = r"D:\PDARS\datasetiso\army worm\16081.jpg"
image2_path = r"D:\PDARS\datasetiso\army worm\16064.jpg"
#compare_images(image1_path, image2_path)