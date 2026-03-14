from roboflow import Roboflow
import os

def download_online_dataset(api_key, workspace_name="balaji-nfzzu", project_name="microplastic_detection-qmdu5", version_number=1):
    """
    Connects to an online dataset provided by Roboflow, and downloads it in the YOLOv8 format 
    (which is fully compatible with YOLO11).
    """
    print(f"Connecting to Roboflow workspace '{workspace_name}'...")
    rf = Roboflow(api_key=api_key)
    
    try:
        project = rf.workspace(workspace_name).project(project_name)
        print(f"Downloading version {version_number} of {project_name}...")
        
        # YOLOv8 format is compatible with YOLO11
        dataset = project.version(version_number).download("yolov8")
        
        print("\n" + "="*50)
        print(f"Dataset successfully downloaded to: {dataset.location}")
        print("="*50)
        print("To use this new dataset for training, make sure the `models/data.yaml`")
        print("points to the paths inside this downloaded folder, then run `train.py` again.")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"An error occurred while trying to download the dataset: {e}")
        print(f"Ensure your API key is correct and that you have access to workspace '{workspace_name}'.")

if __name__ == "__main__":
    print("--- Open Source Microplastics Dataset Downloader ---")
    print("If your model accuracy is low, adding more high-quality diverse images helps!")
    print("You can get a free API key at: https://app.roboflow.com")
    
    api_key_input = input("Enter your Roboflow private API Key: ")
    
    if api_key_input.strip():
        # Using the project listed in the original data.yaml as a default base
        # Users can change these to grab any other public microplastics project.
        download_online_dataset(
            api_key=api_key_input.strip(), 
            workspace_name="balaji-nfzzu", 
            project_name="microplastic_detection-qmdu5", 
            version_number=1
        )
    else:
        print("\nNo API key provided. Exiting script.")
