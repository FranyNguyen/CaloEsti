import os
import shutil
from yolov5 import detect
import glob

weights_path = "D:\\phuong\\Download\\FPTU\\5th_semester\\DPL302m\\Project\\P4\\best.pt"

def object_detection(url): 
    results = detect.run(source=url, imgsz=(512,512), weights=weights_path, conf_thres=0.3)
    
    results_folder = "yolov5/runs/detect/"
    
    exp_folders = glob.glob(os.path.join(results_folder, 'exp*'))
    latest_exp_folder = max(exp_folders, key=os.path.getctime)  

    destination_folder = "static/results/"
    
    os.makedirs(destination_folder, exist_ok=True)

    for file in os.listdir(latest_exp_folder):
        full_file_name = os.path.join(latest_exp_folder, file)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination_folder)
    
    shutil.rmtree(latest_exp_folder)
    
    return f"{destination_folder}result_{os.path.basename(latest_exp_folder)}"


 