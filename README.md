# Microplastic Detection System

This project uses YOLOv11 to detect and quantify microplastics in water samples.

## Project Structure
- `train.py`: Script to train the YOLO model on your dataset. configured for robustness (augmentations, YOLO11s).
- `app.py`: Flask web application for user interaction.
- `utils.py`: Inference logic with TTA and confidence filtering.
- `models/`: Contains the dataset and training configuration.
- `runs/`: Stores training results and model checkpoints.

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Training a Robust Model
To create a robust model customized for your data, run:
```bash
python train.py
```
This will:
- Load the YOLO11s (Small) model.
- Apply data augmentation (rotation, flip, mosaic).
- Train for 100 epochs.
- Save the best model to `runs/detect/train/weights/best.pt`.

## Running the Web App
Start the Flask interface:
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

## Features
- **Robust Training**: Uses augmentation and a larger model backbone.
- **Advanced Inference**: Test Time Augmentation (TTA) and confidence filtering for reliable detection.
- **Web Interface**: Easy-to-use drag & drop interface for analyzing water samples.
"# microplasai" 
"# microplasai" 
"# microplasai" 
"# zoro" 
