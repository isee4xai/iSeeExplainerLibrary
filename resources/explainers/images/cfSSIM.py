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
import traceback

class SSIMCounterfactual(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def nn_data(self,label_raw, label,model_info, data_file,output_names,sample=None):
        train_data = []
        actual=[]
        
        if type(data_file)==str and os.path.isdir(data_file):
            # classification image dataset in zipped folder
            _folders = [_f for _f in os.listdir(data_file) if _f != label_raw]
            if len(_folders)<1:
                raise Exception("No data found.")

            for folder in _folders:
                _folder_path = os.path.join(data_file, folder)
                _files = [os.path.join(_folder_path, f) for f in os.listdir(_folder_path)]
                train_data=train_data+[np.array(Image.open(f)) for f in _files]
                actual=actual+([output_names.index(folder)]*len(_files))
            train_data=np.asarray(train_data)
            actual=np.asarray(actual)

            if sample!=None:
                sample_idx=np.random.randint(train_data.shape[0], size=min(sample,len(train_data)))
                train_data=train_data[sample_idx,:]
                actual=actual[sample_idx]
            train_data = normalise_image_batch(train_data, model_info)

            return train_data,actual
           
        else:
            header = next(data_file).split(',')
            header = [elem.strip() for elem in header]

            while True:
                try:
                    s_instance = next(data_file)
                    s_instance = s_instance.replace('\n', '')
                    s_array = s_instance.split(',')

                    if label != float(s_array[-1]):
                        s_array = [float(s) for s in s_array][:-1]
                        actual.append(float(s_array[-1]))
                        train_data.append(s_array)
                except Exception as e: #end of rows
                    train_data=np.asarray(train_data)
                    actual=np.asarray(actual)

                    if sample!=None:
                        sample_idx=np.random.randint(train_data.shape[0], size=min(sample,len(train_data)))
                        train_data=train_data[sample_idx,:]
                        actual=actual[sample_idx]

                    train_data = train_data.reshape((train_data.shape[0],)+tuple(model_info["attributes"]["features"]["image"]["shape"]))
                    return train_data,actual                
                   
 
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

        return self.explain(_id, instance, params_json)
    

    def nun(self,num_cf,query,data,labels,channel_axis):
        print(query.shape)
        print(data[0].shape)
        similarities=np.array([structural_similarity(query,image,channel_axis=channel_axis) for image in data])
        ind = np.argpartition(similarities, -num_cf)[-num_cf:]
        print(ind)
        top=ind[np.argsort(similarities[ind])[::-1]]

        return top,labels[top]

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
            if(len(pred.shape)==1):
                instance_label = int(np.argmax(pred))
                instance_prob=pred[instance_label]
            else:
                instance_label=pred
                instance_prob=instance_label
            instance_label_raw = output_names[instance_label]

            instance=instance[0]
            
            num_cf = int(params_json["num_cf"]) if "num_cf" in params_json else 2

            sample=None
            if "samples" in params_json:
                sample=int(params_json["samples"])

            channel_axis=None
            if(instance.shape[-1]==3):
                channel_axis=-1

            train_data,labels = self.nn_data(instance_label_raw, instance_label, model_info, data_file,output_names,sample=sample)
            print(len(train_data))
            print(labels)
            cf_indices,cf_labels = self.nun(num_cf,instance,train_data,labels,channel_axis)
            print(cf_indices,cf_labels)
            cf_indices = np.array([train_data[n] for n in cf_indices])
            

            preds=np.asarray(predic_func(cf_indices))
            print(preds)
            if(len(preds.shape)==2):
                preds=preds.tolist()
                print(preds)
                for i in range(cf_indices.shape[0]):
                    preds[i] = [preds[i][cf_labels[i]]]
                preds=np.asarray(preds)
            
            preds=np.squeeze(preds,axis=-1)
            print(preds)

            cf_indices = denormalise_image_batch(cf_indices, model_info)

            size=(12, 6)
            if "png_height" in params_json and "png_width" in params_json:
                try:
                    size=(int(params_json["png_width"])/100.0,int(params_json["png_height"])/100.0)
                except:
                    print("Could not convert dimensions for .PNG output file. Using default dimensions.")

            fig, axes = plt.subplots(nrows=1, ncols=cf_indices.shape[0]+1, figsize=size)
            axes[0].imshow(Image.fromarray(instance_raw))
            axes[0].set_title("Original Image\nClass: " + instance_label_raw+"\nPrediction: "+str(round(instance_prob,4)))
            cf_indices = np.squeeze(cf_indices, axis=3) if cf_indices.shape[-1] == 1 else cf_indices

            print(cf_indices.shape)
            for i in range(cf_indices.shape[0]):
                axes[i+1].imshow(Image.fromarray(cf_indices[i]))
                axes[i+1].set_title("Counterfactual "+str(i+1) +"\nClass: " + output_names[cf_labels[i]]+"\nPrediction: " +str(round(preds[i],3)))
        
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
                "samples":{
                    "description": "Number of samples to use from the background data. The whole dataset is used by default.",
                    "type":"int",
                    "default": None,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "0":"This explanation presents nearest neighbours to the query; nearest neighbours are examples that are similar to the query with similar AI system outcomes."
            },

        "meta":{
                "modelAccess":"File",
                "supportsBWImage":True,
                "needsTrainingData": True


        }
    }
