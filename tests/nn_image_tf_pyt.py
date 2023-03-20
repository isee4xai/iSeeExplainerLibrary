from resources.explainers.images.nn import NearestNeighboursImage
import os
import numpy as np
from utils.base64 import bw_vector_to_base64

data_folder = '/Users/anjanawijekoon/projects/isee/ExplainerLibraries-aw/Models/RADIOGRAPH/RADIOGRAPH.csv'

nn = NearestNeighboursImage("/Users/anjanawijekoon/projects/isee/ExplainerLibraries-aw/Models/", 
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
        #converting to vector
        try:
            instance = bw_vector_to_base64(instance)
        except Exception as e:  
            print("Could not convert vector Image to base64. ", e)
        
print(nn.explain("RADIOGRAPH", instance, {}))
