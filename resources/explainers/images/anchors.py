from http.client import BAD_REQUEST
from re import A
from flask_restful import Resource
from flask import request
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import h5py
import joblib
import json
import matplotlib.pyplot as plt
from alibi.explainers import AnchorImage
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import base64_to_vector,PIL_to_base64
from utils.img_processing import normalize_img
from io import BytesIO
import requests
import traceback


class AnchorsImage(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def post(self):
        try:          
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
            inst_type=params["type"]
            url=None
            if "url" in params:
                url=params["url"]
            params_json={}
            if "params" in params:
                params_json=params["params"]
        
            output_names=None
            predic_func=None
        
            #Getting model info, data, and file from local repository
            model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

            ## params from info
            model_info=json.load(model_info_file)
            backend = model_info["backend"] 
            output_names=model_info["attributes"]["features"][model_info["attributes"]["target_names"][0]]["values_raw"]

            if model_file!=None:
                if backend in ontologyConstants.TENSORFLOW_URIS:
                    model=h5py.File(model_file, 'w')
                    mlp = tf.keras.models.load_model(model)
                    predic_func=mlp.predict
                elif backend in ontologyConstants.SKLEARN_URIS:
                    mlp = joblib.load(model_file)
                    predic_func=mlp.predict_proba
                elif backend in ontologyConstants.PYTORCH_URIS:
                    mlp = torch.load(model_file)
                    predic_func=mlp.predict
                else:
                    mlp = joblib.load(model_file)
                    predic_func=mlp.predict
            elif url!=None:
                def predict(X):
                    return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
                predic_func=predict
            else:
                return "Either a locally stored model or a URL for the prediction function of the model must be provided.",BAD_REQUEST
                

            #converting to vector
            try:
                instance=base64_to_vector(instance)
            except Exception as e:  
                return "Could not convert base64 Image to vector: " + str(e),BAD_REQUEST

            #normalizing
            try:
                instance=normalize_img(instance,model_info)
            except Exception as e:
                    return  "Could not normalize instance: " + str(e),BAD_REQUEST

            if len(model_info["attributes"]["features"]["image"]["shape_raw"])==2 or model_info["attributes"]["features"]["image"]["shape_raw"][-1]==1:
                plt.gray()


            segmentation_fn='slic'
            if "segmentation_fn" in params_json:
                segmentation_fn = params_json["segmentation_fn"]

            segmentation_kwargs={}
            if segmentation_fn=='slic':
                segmentation_kwargs["n_segments"]=10
                if "n_segments" in params_json:
                    segmentation_kwargs["n_segments"]=int(params_json["n_segments"])
            elif segmentation_fn=='quickshift':
                segmentation_kwargs["kernel_size"]=5
                if "kernel_size" in params_json:
                    segmentation_kwargs["kernel_size"]=float(params_json["kernel_size"])
            elif segmentation_fn=='felzenszwalb':
                segmentation_kwargs["scale"]=1
                if "scale" in params_json:
                    segmentation_kwargs["scale"]=float(params_json["scale"])


            threshold=0.95
            if "threshold" in params_json:
                threshold= float(params_json["threshold"])

            delta=0.1
            if "delta" in params_json:
                delta= float(params_json["delta"])

            size=(4, 4)
            if "png_height" in params_json and "png_width" in params_json:
                try:
                    size=(int(params_json["png_width"])/100.0,int(params_json["png_height"])/100.0)
                except:
                    print("Could not convert dimensions for .PNG output file. Using default dimensions.")

            explainer = AnchorImage(predic_func, instance.shape[1:], segmentation_fn=segmentation_fn,segmentation_kwargs=segmentation_kwargs)
            explanation = explainer.explain(instance[0],threshold,delta=delta)

            print(dir(explanation))
        
            fig, axes = plt.subplots(1,1, figsize = size)
            print(explanation.anchor)
            axes.imshow(explanation.anchor)
            if output_names!=None:
                axes.set_title('Predicted Class: '+ output_names[explanation.raw["prediction"][0]] + "\nPrecision: " + str(round(explanation.precision,3)) + "\nCoverage: " + str(round(explanation.coverage,3)))
            else:
                axes.set_title('Predicted Class: '+ str(explanation.raw["prediction"][0]) + "\nPrecision: " + str(round(explanation.precision,3)) + "\nCoverage: " + str(round(explanation.coverage,3)))
        
            #saving
            img_buf = BytesIO()
            plt.savefig(img_buf, bbox_inches='tight',format='png')
            im = Image.open(img_buf)
            b64Image=PIL_to_base64(im)

            response={"type":"image","explanation":b64Image}#,"explanation":json.loads(explanation.to_json())}
            return response
        except:
            return traceback.format_exc(), 500

    def get(self,id=None):
        return {
        "_method_description": "Uses anchors to find the groups of pixels that are sufficient for the model to justify the predicted class."
                           "This method accepts 5 arguments: " 
                           "the 'id', the 'url' (optional),  the 'params' dictionary (optional) with the configuration parameters of the method, the 'instance' containing the image that will be explained as a matrix, or the 'image' file instead. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (images represented as arrays). It must return N outputs (predictions for each image).",
        "instance": "Matrix representing the image to be explained.",
        "image": "Image file to be explained. Ignored if 'instance' was specified in the request. Passing a file is only recommended when the model works with black and white images, or color images that are RGB-encoded using integers ranging from 0 to 255.",
        "params": { 
                "threshold": {
                    "description": "The minimum level of precision required for the anchor. Default is 0.95",
                    "type":"float",
                    "default": 0.95,
                    "range":[0,1],
                    "required":False
                    },
                "delta": {
                    "description": "Confidence threshold for the precision of the anchor. Default is 0.1",
                    "type":"float",
                    "default": 0.1,
                    "range":[0.00001,1],
                    "required":False
                    },
                "segmentation_fn": {
                    "description":"A string with an image segmentation algorithm from the following:'quickshift', 'slic', or 'felzenszwalb'.",
                    "type":"string",
                    "default": "slic",
                    "range":['slic','quickshift','felzenszwalb'],
                    "required":False
                    },
                "n_segments":{
                    "description": "Only used for slic segmentation algorithm. Specifies the (approximate) number of segments of the image.",
                    "type":"int",
                    "default": 10,
                    "range":None,
                    "required":False
                    },
                "kernel_size":{
                    "description": "Only used for quickshift segmentation algorithm. Higher means fewer clusters",
                    "type":"float",
                    "default": 5,
                    "range":None,
                    "required":False
                    },
                "scale":{
                    "description": "Only used for felzenszwalb segmentation algorithm. Higher scale means less and larger segments",
                    "type":"float",
                    "default": 1,
                    "range":None,
                    "required":False
                    },
                "png_width":{
                    "description": "Width (in pixels) of the png image containing the explanation.",
                    "type":"int",
                    "default": 400,
                    "range":None,
                    "required":False
                    },
                "png_height": {
                    "description": "Height (in pixels) of the png image containing the explanation.",
                    "type":"int",
                    "default": 400,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "anchor_image":"Displays the pixels that are sufficient for the model to justify the predicted class."
            },

        "meta":{
                "modelAccess":"Any",
                "supportsBWImage":True,
                "needsTrainingData": False

            }

        }