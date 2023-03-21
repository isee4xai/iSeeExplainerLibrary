from resources.explainers.images.nn import NearestNeighboursImage
import os
import numpy as np
import json
from utils.base64 import vector_to_base64
from utils.img_processing import denormalize_img
from PIL import Image

data_folder = "data-folder"
model_info_path="model-info-json-path"
nn = NearestNeighboursImage("path", "path", data_folder)


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
      
elif os.path.isdir(data_folder):
    _folders = [f for f in os.listdir(data_folder) if f != '.DS_Store']
    if len(_folders)==0:
        raise Exception("No data found.")
    temp = np.random.randint(len(_folders))
    print(temp)
    _folder_path = os.path.join(data_folder, _folders[temp])
    _files = [os.path.join(_folder_path, f) for f in os.listdir(_folder_path)]
    temp = np.random.randint(len(_files))
    print(temp)
    instance = np.array(Image.open(_files[temp]))
    try:
        base_instance = vector_to_base64(instance)
    except Exception as e:  
        print("Could not convert vector Image to base64. ", e)

nn.explain("model-id", base_instance, {})