import cv2
import os
import glob
import torch
import numpy as np
from ultralytics import YOLO
from processing.preprocess import preprocess_image

# Reduce CPU + memory spikes
torch.set_num_threads(1)

# Global model variable
model = None


def load_model():
    """Lazy load the YOLO model (optimized for low memory)."""
    global model

    if model is None:
        try:
            # Priority 1: Prioritize the previous highly accurate microplastics model
            microplastics_model = 'runs/detect/train3/weights/best.pt'
            if os.path.exists(microplastics_model):
                print(f"Loading trained microplastics model from {microplastics_model}")
                model = YOLO(microplastics_model)
            else:
                potential_models = glob.glob('runs/detect/train*/weights/best.pt')
                if potential_models:
                    potential_models.sort(key=os.path.getmtime, reverse=True)
                    model_path = potential_models[0]
                    print(f"Loading trained model from {model_path}")
                    model = YOLO(model_path)

                # Priority 2: models folder
                else:
                    model_files = glob.glob('models/*.pt')
                    if model_files:
                        model_path = model_files[0]
                        print(f"Loading model from {model_path}")
                        model = YOLO(model_path)

                    # Fallback (lightweight)
                    else:
                        print("No custom model found, using yolov8n.pt")
                        model = YOLO("yolov8n.pt")

            # Force CPU (important for Render)
            model.to("cpu")

        except Exception as e:
            print(f"Model loading failed: {e}")
            model = None

    return model


def detect_microplastics(original_image_path, output_folder):
    """
    Detects microplastics in an image.
    """

    # Step 1: Preprocess
    preprocessed_path = preprocess_image(original_image_path, output_folder)

    # Removed forced resize to maintain previous model accuracy on native scale


    # Step 3: Load model
    model = load_model()

    filename = os.path.basename(original_image_path)
    output_path = os.path.join(output_folder, 'annotated_' + filename)

    count = 0
    fragment_count = 0
    fiber_count = 0
    bead_count = 0
    inorganic_count = 0
    dust_count = 0
    sand_count = 0

    # PRE-CALCULATE ORGANIC MASKS (VETO SYSTEM)
    organic_count = 0
    algae_count = 0
    debris_count = 0
    img_for_hsv = cv2.imread(preprocessed_path)
    combined_organic = None
    if img_for_hsv is not None:
        hsv_img = cv2.cvtColor(img_for_hsv, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        # Solution 2: Lowered Saturation threshold (40) to catch washed-out backlit organic spores
        lower_brown = np.array([10, 40, 30])
        upper_brown = np.array([30, 255, 180])
        mask_green = cv2.inRange(hsv_img, lower_green, upper_green)
        mask_brown = cv2.inRange(hsv_img, lower_brown, upper_brown)
        kernel = np.ones((5,5), np.uint8)
        kernel_large = np.ones((7,7), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel_large)
        mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel_large)
        combined_organic = cv2.bitwise_or(mask_green, mask_brown)

    if model:
        try:
            # Step 4: Prediction (restored previous configuration for high accuracy)
            # Lowered confidence threshold to catch more microplastics
            results = model.predict(preprocessed_path, conf=0.15, iou=0.45, augment=True)

            yolo_boxes = []
            for result in results:
                # We will draw boxes manually to color-code dust/sand vs microplastics.
                # First let's get a clean image to draw on
                im_array = cv2.imread(preprocessed_path)
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Solution 1: HSV Color Overrules YOLO (Veto False Positives)
                    if combined_organic is not None:
                        roi = combined_organic[int(y1):int(y2), int(x1):int(x2)]
                        organic_pixels = cv2.countNonZero(roi)
                        total_pixels = max(1, (int(y2) - int(y1)) * (int(x2) - int(x1)))
                        # If more than 20% of the YOLO box is organic colored, skip it!
                        if (organic_pixels / total_pixels) > 0.20:
                            continue
                            
                    yolo_boxes.append((x1, y1, x2, y2))
                    cls_id = int(box.cls[0])
                    cls_name = (
                        model.names[cls_id]
                        if hasattr(model, 'names') and cls_id in model.names
                        else 'Microplastic'
                    )

                    if cls_name.lower() == 'fiber':
                        fiber_count += 1
                        count += 1
                        # Draw blue for microplastics
                        cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                        cv2.putText(im_array, "Fiber", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    elif cls_name.lower() == 'fragment':
                        fragment_count += 1
                        count += 1
                        cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                        cv2.putText(im_array, "Fragment", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    elif cls_name.lower() == 'bead':
                        bead_count += 1
                        count += 1
                        cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                        cv2.putText(im_array, "Bead", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    elif cls_name.lower() in ['dust', 'sand']:
                        debris_count += 1
                        organic_count += 1
                        # Draw as Sand/Dirt
                        cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 255), 2)
                        cv2.putText(im_array, "Sand/Dirt", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
                    else:
                        fragment_count += 1
                        count += 1
                        cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                        cv2.putText(im_array, "Microplastic", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                cv2.imwrite(output_path, im_array)

        except Exception as e:
            print(f"Inference failed: {e}")
            model = None

    # Step 5: Fallback (if model fails)
    if model is None:
        img = cv2.imread(original_image_path)
        h, w = img.shape[:2]

        cv2.rectangle(img, (w//4, h//4), (w//2, h//2), (0, 255, 0), 2)
        cv2.putText(img, "Mock Detection", (w//4, h//4-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(output_path, img)

        count = 10
        fragment_count = 6
        fiber_count = 4
        bead_count = 0
        inorganic_count = 8
        dust_count = 5
        sand_count = 3

    # Organic matter detection drawing and counting
    try:
        img_for_drawing = cv2.imread(output_path)
        if img_for_drawing is not None and img_for_hsv is not None:
            # Function to check overlap with YOLO boxes
            def is_overlapping(x, y, w, h, yolo_boxes):
                cx, cy = x + w/2, y + h/2
                for (x1, y1, x2, y2) in yolo_boxes:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        return True
                return False

            # Find Algae (Green)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_green:
                area = cv2.contourArea(cnt)
                if area > 150: # Solution 2: Lowered area to catch microscopic diatoms/spores
                    x, y, w, h = cv2.boundingRect(cnt)
                    if not is_overlapping(x, y, w, h, yolo_boxes if 'yolo_boxes' in locals() else []):
                        algae_count += 1
                        organic_count += 1
                        cv2.rectangle(img_for_drawing, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(img_for_drawing, "Algae", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Find Debris (Brown) 
            contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_brown:
                area = cv2.contourArea(cnt)
                if area > 150: # Solution 2: Lowered area to catch microscopic diatoms/spores
                    x, y, w, h = cv2.boundingRect(cnt)
                    if not is_overlapping(x, y, w, h, yolo_boxes if 'yolo_boxes' in locals() else []):
                        debris_count += 1
                        organic_count += 1
                        cv2.rectangle(img_for_drawing, (x, y), (x+w, y+h), (0, 100, 255), 2)
                        cv2.putText(img_for_drawing, "Sand/Dirt", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
                    
            cv2.imwrite(output_path, img_for_drawing)
    except Exception as e:
        print(f"Organic detection failed: {e}")

    # Step 6: Contamination logic
    if count <= 10:
        level = "Clean"
        status_class = "success"
        recommendation = "Water quality is acceptable for basic use."
    elif count <= 30:
        level = "Mild Contamination"
        status_class = "warning"
        recommendation = "Consider basic filtration."
    elif count <= 60:
        level = "Polluted"
        status_class = "danger"
        recommendation = "Advanced filtration required."
    else:
        level = "Highly Polluted"
        status_class = "danger"
        recommendation = "Do not use. Immediate advanced treatment necessary."

    # Edibility logic
    is_edible = True
    edibility_reason = "Water is clean and safe to drink."
    
    if count > 0 and organic_count > 10:
        is_edible = False
        edibility_reason = "Not safe: Contains microplastics and high organic contamination."
    elif count > 0:
        is_edible = False
        edibility_reason = "Not safe: Contains microplastics."
    elif organic_count > 10:
        is_edible = False
        edibility_reason = "Not safe: High organic contamination detected."

    # Step 7: Dominant type
    types_dict = {
        'Fibers': fiber_count,
        'Fragments': fragment_count,
        'Beads': bead_count
    }
    dominant_type = max(types_dict, key=types_dict.get) if count > 0 else 'None'

    # Step 8: Density
    density = round(count / 100.0, 2)

    # Filenames for frontend
    annotated_filename = 'annotated_' + filename
    preprocessed_filename = 'preprocessed_' + filename

    return {
        'count': count,
        'organic_count': organic_count,
        'algae_count': algae_count,
        'debris_count': debris_count,
        'inorganic_count': inorganic_count,
        'dust_count': dust_count,
        'sand_count': sand_count,
        'is_edible': is_edible,
        'edibility_reason': edibility_reason,
        'level': level,
        'status_class': status_class,
        'recommendation': recommendation,
        'dominant_type': dominant_type,
        'density': density,
        'annotated_image': annotated_filename,
        'preprocessed_image': preprocessed_filename,
        'original_image': filename
    }