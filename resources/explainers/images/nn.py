from flask_restful import Resource
from flask import request
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import tensorflow as tf
import torch
import h5py
import heapq
import json
from io import BytesIO
import csv
import matplotlib.pyplot as plt
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import base64_to_vector, PIL_to_base64, vector_to_base64
from utils.img_processing import normalize_img, normalise_image_batch, denormalise_image_batch
from sklearn.metrics.pairwise import euclidean_distances

class NearestNeighboursImage(Resource):

    def __init__(self,model_folder,upload_folder,data_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        self.data_folder = data_folder
        
    def nn_data(self, label_raw, label, model_info, encoder):
        train_data = []
        if not os.path.exists(self.data_folder):
            raise Exception("No data found.")
        
        if os.path.isdir(self.data_folder):
            # classification image dataset in zipped folder
            _folders = [_f for _f in os.listdir(self.data_folder) if _f == label_raw]
            if len(_folders)!=1:
                raise Exception("No data found.")
            _folder_path = os.path.join(self.data_folder, _folders[0])
            _files = [os.path.join(self._folder_path, f) for f in os.listdir(_folder_path)]
            train_data = [np.array(Image.open(f)) for f in _files]
            train_data = normalise_image_batch(train_data)
            train_encodings = encoder(train_data)
            return train_data, train_encodings
        
        if os.path.isfile(self.data_folder):
            # csv file, first column is column names, 1st column maybe index 
            with open(self.data_folder, 'r') as f:
                header = next(f).split(' ')
                header = [elem.strip() for elem in header]

                while True:
                    try:
                        s_instance = next(f)
                        s_instance = s_instance.replace('\n', '')
                        s_array = s_instance.split(',')
                        if label == float(s_array[-1]):
                            s_array = [float(s) for s in s_array][:-2]
                            train_data.append(s_array)
                    except Exception as e: #end of rows
                        train_data = np.asarray(train_data, dtype=float)
                        train_data = train_data.reshape((train_data.shape[0],)+tuple(model_info["attributes"]["features"]["image"]["shape"]))
                        train_encodings = encoder(train_data)
                        return train_data, train_encodings                     
                    
    def knn(self, sample_size, data, query):
        ecd = euclidean_distances(query, data)[0]
        top = heapq.nsmallest(sample_size+1, range(len(ecd)), ecd.take)
        return top
 
    def post(self):
        params = request.json
        if params is None:
            return "The params are missing"

        #check params
        if("id" not in params):
            return "The model id was not specified in the params."
        if("type" not in params):
            return "The instance type was not specified in the params."
        if("instance" not in params):
            return "The instance was not specified in the params."
        
        _id =params["id"]
        instance = params["instance"]
        params_json={}
        if "params" in params:
            params_json=params["params"]

        return self.explain(_id, instance, params_json)
    
    def explain(self, model_id, instance, params_json):
        no_neighbours = params_json["no_neighbours"] if "no_neighbours" in params_json else 3

        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(model_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        output_names=model_info["attributes"]["features"][model_info["attributes"]["target_names"][0]]["values_raw"]

        predic_func=None
        last_layer_func=None

        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model = h5py.File(model_file, 'w')
                model = tf.keras.models.load_model(model)
                predic_func=model   
                def last_layer(x):
                    new_model = tf.keras.models.Model([model.inputs], [model.layers[-2].output])
                    return new_model(x)
                last_layer_func = last_layer
            elif backend in ontologyConstants.PYTORCH_URIS:
                model = torch.load(model_file)
                predic_func=model.predict
                def last_layer(x):
                    new_model = nn.Sequential(*list(model.children())[:-1])
                    return new_model(x).flatten()
                last_layer_func = last_layer
            else:
                raise Exception("Only Tensorflow and PyTorch backends are supported.")
        else:
            raise Exception("A nerual model must be provided.")
        
        try:
            instance = base64_to_vector(instance)
        except Exception as e:  
            return "Could not convert base64 Image to vector: " + str(e)

        instance_raw = instance #Raw format needed for explanation

        #normalise and reshape
        try:
            instance=normalize_img(instance,model_info)
        except Exception as e:
                return  "Could not normalize instance: " + str(e)
        instance_label = int(predic_func(instance)[0].numpy().argmax())
        instance_label_raw = output_names[instance_label]
        train_data, train_encodings = self.nn_data(instance_label_raw, instance_label, model_info, last_layer_func)
        nn_indices = self.knn(no_neighbours, train_encodings, last_layer_func(instance))
        print(nn_indices)
        nn_instances = [train_data[n] for n in nn_indices]
        nn_instances = denormalise_image_batch(nn_instances, model_info)
        size=(12, 12)
        if "png_height" in params_json and "png_width" in params_json:
            try:
                size=(int(params_json["png_width"])/100.0,int(params_json["png_height"])/100.0)
            except:
                print("Could not convert dimensions for .PNG output file. Using default dimensions.")

        fig, axes = plt.subplots(1,nn_instances.shape[0]+1, figsize = size)
        axes[0].imshow(Image.fromarray(instance_raw))
        nn_instances = np.squeeze(nn_instances, axis=3) if nn_instances.shape[3] == 1 else nn_instances
        for i in range(nn_instances.shape[0]):
            axes[i+1].imshow(Image.fromarray(nn_instances[i]))
        fig.savefig("tests/test.png")
        
        #saving
        img_buf = BytesIO()
        fig.savefig(img_buf)
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response={"type":"image","explanation":b64Image}
        return response

    def get(self):
        return {
        "_method_description": "",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Image to be explained in BASE64 format",
        "params": { 
                "no_neighbours": "(Optional) number of neighbours returned as an interger; default is 3",
                },
        "output_description":{
                "0":"This explanation presents nearest neighbours to the query; nearest neighbours are examples that are similar to the query with similar AI system outcomes."
            },
        "meta":{
                "supportsAPI":True,
                "supportsB&WImage":False,
                "needsData": False,
                "requiresAttributes":[]
            }
        }