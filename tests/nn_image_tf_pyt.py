from resources.explainers.images.nn import NearestNeighboursImage
import os
import numpy as np
import json
from utils.base64 import vector_to_base64
from utils.img_processing import denormalize_img

data_folder = "/Users/anjanawijekoon/projects/isee/iSeeExplainerLibrary-aw/Models/RADIOGRAPH/RADIOGRAPH.csv"
model_info_path="/Users/anjanawijekoon/projects/isee/iSeeExplainerLibrary-aw/Models/RADIOGRAPH/RADIOGRAPH.json"
nn = NearestNeighboursImage("/Users/anjanawijekoon/projects/isee/iSeeExplainerLibrary-aw/Models/", 
                       "",
                       data_folder)


if os.path.isfile(data_folder):
    # csv file, first column is column names, 1st column maybe index 
    with open(data_folder, 'r') as f:
        header = next(f).split(',')
        header = [elem.strip() for elem in header]
        temp = np.random.randint(30)
        print(temp)
        for i in range(temp):
            s_instance = next(f)
        s_instance = next(f)
        s_instance = s_instance.replace('\n', '')
        s_array = s_instance.split(',')
        s_array = [float(s) for s in s_array][:-2]
        instance = np.array(s_array)
        model_info=None
        with open(model_info_path, 'r') as info_file:
            model_info=json.load(info_file)
        try:
            instance=denormalize_img(instance,model_info)
        except Exception as e:  
            print("Could not denormalize image. ", e)
        try:
            instance = vector_to_base64(instance)
        except Exception as e:  
            print("Could not convert vector Image to base64. ", e)
      
nn.explain("RADIOGRAPH", instance, {})