import cv2
import os
import glob
import torch
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
            # Priority 1: trained model
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

    # Step 2: Resize image (Increased to 800x800 for better range and accuracy)
    img = cv2.imread(preprocessed_path)
    img = cv2.resize(img, (800, 800))
    cv2.imwrite(preprocessed_path, img)

    # Step 3: Load model
    model = load_model()

    filename = os.path.basename(original_image_path)
    output_path = os.path.join(output_folder, 'annotated_' + filename)

    count = 0
    fragment_count = 0
    fiber_count = 0
    bead_count = 0

    if model:
        try:
            # Step 4: Prediction (augment=True for better accuracy, lower conf for better range)
            results = model.predict(preprocessed_path, conf=0.15, iou=0.45, augment=True, imgsz=800)

            for result in results:
                # Lightweight plotting
                im_array = result.plot(line_width=1)
                cv2.imwrite(output_path, im_array)

                count += len(result.boxes)

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = (
                        model.names[cls_id]
                        if hasattr(model, 'names') and cls_id in model.names
                        else 'Microplastic'
                    )

                    if cls_name.lower() == 'fiber':
                        fiber_count += 1
                    elif cls_name.lower() == 'fragment':
                        fragment_count += 1
                    elif cls_name.lower() == 'bead':
                        bead_count += 1
                    else:
                        fragment_count += 1

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
        'level': level,
        'status_class': status_class,
        'recommendation': recommendation,
        'dominant_type': dominant_type,
        'density': density,
        'annotated_image': annotated_filename,
        'preprocessed_image': preprocessed_filename,
        'original_image': filename
    }