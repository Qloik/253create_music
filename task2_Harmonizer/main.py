import os
from data_processing import create_folders, convert_xml_to_json
from train import train_baseline_model, train_improved_model
from generate import test_and_generate_baseline, test_and_generate_enhanced
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from music21 import corpus, chord, note, converter, stream, meter, key, tempo, duration
from data_processing import Config

# ===== Main function =====
def main():
    print("=== Four-Part Choir Generation System ===")
    print("1. Convert XML to JSON")
    print("2. Train Model")
    print("3. Test and Generate")
    print("4. Full Pipeline")
    
    choice = input("Choose an option (1-4): ").strip()
    create_folders()
    
    if choice == "1":
        convert_xml_to_json()
        
    elif choice == "2":
        if not os.path.exists(Config.RAW_JSON):
            print(f"Data file {Config.RAW_JSON} doesn't exist, please run XML to JSON conversion first")
            return
        
        print("\n=== Training Options ===")
        print("2a. Train Baseline Model")
        print("2b. Train Improved Model")
        
        train_choice = input("Choose training option (2a/2b): ").strip().lower()
        
        if train_choice == "2a":
            print("Training baseline model...")
            train_baseline_model()
        elif train_choice == "2b":
            print("Training improved model...")
            train_improved_model()
        else:
            print("Invalid training choice")
            
    elif choice == "3":
        print("\n=== Generation Options ===")
        print("3a. Test Baseline Model")
        print("3b. Test Enhanced Model") 
        
        test_choice = input("Choose testing option (3a/3b): ").strip().lower()
        
        if test_choice == "3a":
            # Check if baseline model exists
            baseline_paths = Config.BASELINE_MODEL_PATH    
            model_found = False
            
            if os.path.exists(baseline_paths):
                model_found = True
            
            if not model_found:
                print("No baseline model found. Please train a model first.")
                return
                
            print("Testing baseline model...")
            test_and_generate_baseline()
            
        elif test_choice == "3b":
            # Check if improved model exists
            improved_paths =  Config.IMPROVED_MODEL_PATH  
            model_found = False
         
            if os.path.exists(improved_paths):
                model_found = True
               
            if not model_found:
                print("No improved model found. Please train a model first.")
                return
                
            print("Testing enhanced model...")
            test_and_generate_enhanced()

        else:
            print("Invalid testing choice")
            
    elif choice == "4":
        print("\n=== Full Pipeline Options ===")
        print("4a. Full Pipeline - Baseline Model")
        print("4b. Full Pipeline - Improved Model")
        
        pipeline_choice = input("Choose pipeline option (4a/4b): ").strip().lower()
        
        print("Running full pipeline...")
        
        if convert_xml_to_json():
            if pipeline_choice == "4a":
                print("\n--- Full Baseline Pipeline ---")
                train_baseline_model()
                test_and_generate_baseline()
                
            elif pipeline_choice == "4b":
                print("\n--- Full Improved Pipeline ---")
                train_improved_model()
                test_and_generate_enhanced()
            else:
                print("Invalid pipeline choice")
                return
                
            print("\nFull pipeline completed successfully!")
        else:
            print("XML conversion failed. Pipeline stopped.")
            
    else:
        print("Invalid choice. Please select 1-4.")

def show_model_status():
    """Show status of available models"""
    print("\n=== Model Status ===")
    
    baseline_paths =  Config.BASELINE_MODEL_PATH
    improved_paths = Config.IMPROVED_MODEL_PATH
    
    # Check baseline models
    baseline_found = False
    for path in baseline_paths:
        if os.path.exists(path):
            print(f" Baseline model found: {path}")
            baseline_found = True
            break
    
    if not baseline_found:
        print(" No baseline model found")
    
    # Check improved models  
    improved_found = False
    for path in improved_paths:
        if os.path.exists(path):
            print(f" Improved model found: {path}")
            improved_found = True
            break
    
    if not improved_found:
        print(" No improved model found")
    
    # Check data
    if os.path.exists(Config.RAW_JSON):
        print(f" Training data found: {Config.RAW_JSON}")
    else:
        print(f" No training data found: {Config.RAW_JSON}")
    
    print()

if __name__ == "__main__":
    # Show model status first
    show_model_status()
    main()