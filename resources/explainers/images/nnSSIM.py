from http.client import BAD_REQUEST
from flask_restful import Resource
from flask import request
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import torch
import h5py
import json
import matplotlib.pyplot as plt
from io import BytesIO
from skimage.metrics import structural_similarity
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import base64_to_vector, PIL_to_base64
from utils.img_processing import normalize_img, normalise_image_batch, denormalise_image_batch
from utils.img_processing import normalise_image_batch
from utils.validation import validate_params
import traceback

class SSIMNearestNeighbours(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def nn_data(self,label_raw, label,model_info, data_file,sample=None):
        train_data = []

        
        if type(data_file)==str and os.path.isdir(data_file):
            # classification image dataset in zipped folder
            _folders = [_f for _f in os.listdir(data_file) if _f == label_raw]
            if len(_folders)<1:
                raise Exception("No data found.")

            _folder_path = os.path.join(data_file, _folders[0])
            _files = [os.path.join(_folder_path, f) for f in os.listdir(_folder_path)]
            train_data=train_data+[np.array(Image.open(f)) for f in _files]
            train_data=np.asarray(train_data)

            if sample!=None:
                sample_idx=np.random.randint(train_data.shape[0], size=min(sample,len(train_data)))
                train_data=train_data[sample_idx,:]
            train_data = normalise_image_batch(train_data, model_info)

            return train_data
           
        else:
            header = next(data_file).split(',')
            header = [elem.strip() for elem in header]

            while True:
                try:
                    s_instance = next(data_file)
                    s_instance = s_instance.replace('\n', '')
                    s_array = s_instance.split(',')
                    if label == float(s_array[-1]):
                        s_array = [float(s) for s in s_array][:-1]
                        train_data.append(s_array)
                except Exception as e: #end of rows
                    train_data=np.asarray(train_data)

                    if sample!=None:
                        sample_idx=np.random.randint(train_data.shape[0], size=min(sample,len(train_data)))
                        train_data=train_data[sample_idx,:]

                    train_data = train_data.reshape((train_data.shape[0],)+tuple(model_info["attributes"]["features"]["image"]["shape"]))
                    return train_data                 
                   
 
    def post(self):
        params = request.json
        if params is None:
            return "The params are missing",BAD_REQUEST

        #check params
        if("id" not in params):
            return "The model id was not specified in the params.",BAD_REQUEST
        if("type" not in params):
            return "The instance type was not specified in the params.",BAD_REQUEST
        if("instance" not in params):
            return "The instance was not specified in the params.",BAD_REQUEST
        
        _id =params["id"]
        instance = params["instance"]
        params_json={}
        if "params" in params:
            params_json=params["params"]
        params_json=validate_params(params_json,self.get(_id)["params"])

        return self.explain(_id, instance, params_json)
    

    def knn(self,no_neighbours,query,data,channel_axis):
        similarities=np.array([structural_similarity(query,image,channel_axis=channel_axis) for image in data])
        ind = np.argpartition(similarities, -(no_neighbours+1))[-(no_neighbours+1):]

        top=ind[np.argsort(similarities[ind])[::-1]]


        return top,similarities[top]

    def explain(self, model_id, instance, params_json):
        try:

            #Getting model info, data, and file from local repository
            model_file, model_info_file, data_file = get_model_files(model_id,self.model_folder)

            ## params from info
            model_info=json.load(model_info_file)
            backend = model_info["backend"]  ##error handling?
            output_names=model_info["attributes"]["features"][model_info["attributes"]["target_names"][0]]["values_raw"]

            predic_func=None

            if model_file!=None:
                if backend in ontologyConstants.TENSORFLOW_URIS:
                    model = h5py.File(model_file, 'w')
                    model = tf.keras.models.load_model(model)
                    predic_func=model   
                elif backend in ontologyConstants.PYTORCH_URIS:
                    model = torch.load(model_file)
                    predic_func=model.predict
                else:
                    return "Only Tensorflow and PyTorch backends are supported.",BAD_REQUEST
            else:
                return "A ML model must be provided.",BAD_REQUEST
        
            try:
                instance = base64_to_vector(instance)
            except Exception as e:  
                return "Could not convert base64 Image to vector: " + str(e),BAD_REQUEST

            instance_raw = instance #Raw format needed for explanation

            #normalise and reshape
            try:
                instance=normalize_img(instance,model_info)
            except Exception as e:
                    return  "Could not normalize instance: " + str(e),BAD_REQUEST

            pred=np.array(predic_func(instance)[0])
            print(pred.shape)
            if(len(pred.shape)==1):
                instance_label = int(np.argmax(pred))
                instance_prob=pred[instance_label]
            else:
                instance_label=pred
                instance_prob=instance_label

            print(instance_label)
            print(instance_prob)
            instance_label_raw = output_names[instance_label]
            print(instance_label_raw)

            instance=instance[0]
            
            no_neighbours = params_json["no_neighbours"]
            sample=params_json["samples"]
  
            channel_axis=None
            if(instance.shape[-1]==3):
                channel_axis=-1

            train_data = self.nn_data(instance_label_raw, instance_label, model_info, data_file,sample=sample)
            nn_indices,sims = self.knn(no_neighbours,instance,train_data,channel_axis)
            nn_instances = np.array([train_data[n] for n in nn_indices[1:]])
            sims=(1+sims[1:])/2
            
            preds=predic_func(nn_instances)

            #if(len(preds.shape)==2):
            #    preds = np.delete(preds,np.s_[0:instance_label],1)
            #    print(preds[0])
            #    preds = np.delete(preds,np.s_[0:-1],1)

            preds=np.squeeze(preds)

            nn_instances = denormalise_image_batch(nn_instances, model_info)

            size=(params_json["png_width"]/100.0,params_json["png_height"]/100.0)

            fig, axes = plt.subplots(nrows=1, ncols=nn_instances.shape[0]+1, figsize=size)
            axes[0].imshow(Image.fromarray(instance_raw))
            axes[0].set_title("Original Image\n Class: " + instance_label_raw+"\nPrediction: "+str(round(instance_prob,3)))
            nn_instances = np.squeeze(nn_instances, axis=3) if nn_instances.shape[-1] == 1 else nn_instances

 
            for i in range(nn_instances.shape[0]):
                axes[i+1].imshow(Image.fromarray(nn_instances[i]))
                axes[i+1].set_title("Neighbour "+str(i+1)+"\nSimilarity: "+str(round(sims[i],3))+"\nPrediction: " + str(round(preds[i][instance_label],3)))
        
            for ax in fig.axes:
                ax.axis('off')
    
            #saving
            img_buf = BytesIO()
            fig.savefig(img_buf,bbox_inches='tight')
            im = Image.open(img_buf)
            b64Image=PIL_to_base64(im)

            response={"type":"image","explanation":b64Image}
            return response
        except:
            return traceback.format_exc(), 500

    def get(self,id=None):
        return {
        "_method_description": "Finds the nearest neighbours to a data instances based on minimum euclidean distance",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Image to be explained in BASE64 format",
        "params": {
                "no_neighbours":{
                        "description": "Number of neighbours to be found. Defaults to 3.",
                        "type":"int",
                        "default": 3,
                        "range":None,
                        "required":False
                        },
                "samples":{
                    "description": "Number of samples to use from the background data. A hundred samples are used by default.",
                    "type":"int",
                    "default": 100,
                    "range":None,
                    "required":False
                    },                
                "png_width":{
                    "description": "Width (in pixels) of the png image containing the explanation.",
                    "type":"int",
                    "default": 1200,
                    "range":None,
                    "required":False
                    },
                "png_height": {
                    "description": "Height (in pixels) of the png image containing the explanation.",
                    "type":"int",
                    "default": 600,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "0":"This explanation presents the nearest neighbours of the query using Structural Similarity Index Measure (SSIM). Nearest neighbours are examples that are similar to the query with similar AI system outcomes."
            },

        "meta":{
                "modelAccess":"File",
                "supportsBWImage":True,
                "needsTrainingData": True


        }
    }
