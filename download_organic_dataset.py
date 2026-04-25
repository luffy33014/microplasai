from roboflow import Roboflow
import os
import yaml

def download_organic_dataset(api_key):
    """
    Connects to Roboflow and downloads an organic water contaminants dataset
    (containing Algae, Organic Debris, and Microplastics).
    Updates models/data.yaml seamlessly.
    """
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    # In a real environment, the user points this to their actual curated project.
    workspace_name = "balaji-nfzzu"
    project_name = "microplastic_detection-qmdu5"
    version_number = 1
    
    try:
        project = rf.workspace(workspace_name).project(project_name)
        print(f"Downloading {project_name} - Version {version_number} [YOLOv8 Format]...")
        
        # Download the dataset
        dataset = project.version(version_number).download("yolov8")
        dataset_path = os.path.abspath(dataset.location)
        
        print("\n" + "="*50)
        print(f"Dataset successfully downloaded to: {dataset_path}")
        print("="*50)
        
        # Update models/data.yaml to point to this new dataset for immediate training
        yaml_path = os.path.abspath("models/data.yaml")
        
        # Modify the yaml dynamically
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                
            # Update paths replacing old static paths with new dataset paths
            data['train'] = os.path.join(dataset_path, "train", "images").replace('\\', '/')
            data['val'] = os.path.join(dataset_path, "valid", "images").replace('\\', '/')
            data['test'] = os.path.join(dataset_path, "test", "images").replace('\\', '/')
            
            # Ensure classes are correct
            data['nc'] = 5
            data['names'] = ['Microplastic', 'Algae', 'Organic Debris', 'Dust', 'Sand']
                
            with open(yaml_path, 'w') as file:
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)
                
            print(f"Successfully updated {yaml_path} with new dataset paths!")
        else:
            print("Could not find models/data.yaml to auto-update. Please update it manually.")
            
        print("You are now ready to run `python train.py` to train the multi-class model.")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"An error occurred while trying to download the dataset: {e}")
        print("Please ensure your API key is correctly typed and has access bounds.")

if __name__ == "__main__":
    print("--- Organic Dataset Downloader ---")
    print("Downloading training data for: Microplastics, Algae, Organic Debris.")
    
    # Executing for user with provided key
    api_key_input = "zPAYKm87ylvC1p94mfzm"
    download_organic_dataset(api_key_input)
