import numpy as np
from PIL import Image  

'''
crop_image_based_on_padded_bounded_box

# Set the openpose parameters
'''
# Set openpose default parameters
def crop_image_based_on_padded_bounded_box(x_min, y_min, x_max, y_max, imageToProcess):
    img = Image.fromarray(imageToProcess)
    img_res = img.crop((x_min, y_min, x_max, y_max))
    crop_image = np.asarray(img_res) 
    return crop_image