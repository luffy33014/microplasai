import cv2
import os
import glob
from ultralytics import YOLO

# Global model variable
model = None

def load_model():
    """Lazy load the YOLO model."""
    global model
    if model is None:
        # Check for trained model in runs folder recursively
        potential_models = glob.glob('runs/detect/train*/weights/best.pt')
        if potential_models:
            # Sort by modification time to get the latest
            potential_models.sort(key=os.path.getmtime, reverse=True)
            model_path = potential_models[0]
            print(f"Loading trained model from {model_path}")
            model = YOLO(model_path)
            # Warmup
            model.fuse()
        else:
            # Check for model file in models directory
            model_files = glob.glob('models/*.pt')
            if model_files:
                model_path = model_files[0]
                print(f"Loading model from {model_path}")
                model = YOLO(model_path)
            else:
                print("No custom model found, using yolo11s.pt as fallback")
                try:
                    model = YOLO("yolo11s.pt")
                except:
                    print("Could not load standard model.")
                    model = None
    return model

def detect_microplastics(image_path, output_folder):
    """
    Detects microplastics in an image.
    
    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder to save the annotated image.
        
    Returns:
        dict: "count", "level", "annotated_image", "status_class"
    """
    model = load_model()
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, 'annotated_' + filename)
    
    count = 0
    
    if model:
        # Run inference with robust thresholds
        # conf: Confidence threshold
        # iou: NMS IoU threshold
        # augment: TTA (Test Time Augmentation) for better robustness at cost of speed
        results = model.predict(image_path, conf=0.25, iou=0.45, augment=True)
        
        # Visualize the results
        for result in results:
            # save=True plots it to a run folder, but we want to save specifically
            # plot() returns a numpy array (BGR)
            im_array = result.plot(line_width=2, font_size=1.0)
            cv2.imwrite(output_path, im_array)
            
            # Count boxes
            count += len(result.boxes)
            
    else:
        # Mock behavior if no model is present
        print("Mocking detection...")
        img = cv2.imread(image_path)
        # Draw some dummy rectangles
        h, w = img.shape[:2]
        cv2.rectangle(img, (w//4, h//4), (w//2, h//2), (0, 255, 0), 2)
        cv2.putText(img, "Mock Detection", (w//4, h//4-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(output_path, img)
        count = 42 # Dummy count

    # Determine contamination level
    if count == 0:
        level = "Clean"
        status_class = "success"
    elif count < 10:
        level = "Low"
        status_class = "warning"
    elif count < 50:
        level = "Moderate"
        status_class = "warning"
    else:
        level = "Polluted"
        status_class = "danger"

    return {
        'count': count,
        'level': level,
        'status_class': status_class,
        'annotated_image': 'annotated_' + filename
    }
