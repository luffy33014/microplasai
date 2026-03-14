from ultralytics import YOLO
import os

def train_model():
    # Load a larger model for better accuracy
    # Using yolo11m.pt (Medium) instead of Small for higher accuracy on small objects like microplastics
    print("Loading YOLO11 Medium model...")
    model = YOLO("yolo11m.pt") 

    # Train the model
    # We use the absolute path to data.yaml to avoid issues
    data_path = os.path.abspath("models/data.yaml")
    
    print(f"Starting training with data config: {data_path}")
    
    # highly optimized robust training configuration to increase accuracy
    try:
        results = model.train(
            data=data_path,
            epochs=150,           # Increased epochs for longer learning
            imgsz=800,            # Increased image size (crucial for small microplastics)
            plots=True,
            batch=16,             # Adjust based on GPU memory. If OOM occurs, reduce to 8
            patience=30,          # Longer early stopping patience
            optimizer='AdamW',    # AdamW optimizer often works better for object detection
            lr0=0.001,            # Initial learning rate
            weight_decay=0.0005,  # Regularization
            
            # Heavy Augmentations to prevent overfitting and improve robustness
            degrees=15.0,         # Rotation
            translate=0.15,       # Translation
            scale=0.6,            # Scaling
            shear=2.5,            # Shear
            flipud=0.5,           # Flip up-down (water samples have no fixed orientation)
            fliplr=0.5,           # Flip left-right
            mosaic=1.0,           # Mosaic augmentation (great for small objects)
            mixup=0.15,           # Mixup
            copy_paste=0.1,       # Copy-paste augmentation (very effective for small scattered objects)
            auto_augment='randaugment', # Leverage auto-augmentation policies
        )
        print("Training completed successfully.")
        
        # After training, the best model will be saved in runs/detect/train/weights/best.pt
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
        print("Note: If the accuracy is still low, try running the new `download_extra_dataset.py` script to get more data!")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    train_model()
