import cv2
import numpy as np
import os

def preprocess_image(image_path, output_folder):
    """
    OpenCV preprocessing pipeline for microplastic images.
    - Convert to HSV/RGB
    - Noise reduction (Gaussian Blur)
    - Contrast enhancement (CLAHE)
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path
        
    # Noise reduction
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Contrast enhancement using CLAHE on L channel of LAB color space
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Highlight fluorescent particles (optional, simple color manipulation)
    # converting to HSV to increase saturation slightly
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 20)
    v = cv2.add(v, 10)
    hsv_enhanced = cv2.merge((h, s, v))
    final_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    filename = os.path.basename(image_path)
    preprocessed_path = os.path.join(output_folder, 'preprocessed_' + filename)
    cv2.imwrite(preprocessed_path, final_img)
    
    return preprocessed_path
