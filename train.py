from ultralytics import YOLO
import os

def train_model():
    # Load a model
        # Using yolo11s.pt (Small) instead of Nano for better accuracy on small objects like microplastics
    model = YOLO("yolo11s.pt") 

    # Train the model
    # We use the absolute path to data.yaml to avoid issues
    data_path = os.path.abspath("models/data.yaml")
    
    print(f"Starting training with data config: {data_path}")
    
    # robust training configuration
    try:
        results = model.train(
            data=data_path,
            epochs=100,           # Increased epochs
            imgsz=640,
            plots=True,
            batch=16,             # Adjust based on GPU memory
            patience=20,          # Early stopping
            
            # Augmentations for robustness
            degrees=10.0,         # Rotation
            translate=0.1,        # Translation
            scale=0.5,            # Scaling
            shear=2.0,            # Shear
            flipud=0.5,           # Flip up-down (water samples have no fixed orientation)
            fliplr=0.5,           # Flip left-right
            mosaic=1.0,           # Mosaic augmentation
            mixup=0.1,            # Mixup
        )
        print("Training completed successfully.")
        
        # After training, the best model will be saved in runs/detect/train/weights/best.pt
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    train_model()
