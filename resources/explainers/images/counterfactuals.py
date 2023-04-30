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
from alibi.explainers import Counterfactual
from getmodelfiles import get_model_files
import requests
from io import BytesIO
from utils import ontologyConstants
from utils.base64 import base64_to_vector,PIL_to_base64
from utils.img_processing import normalize_img

BACKENDS=["TF1",
	"TF2",
	"TF",
    "TensorFlow1",
    "TensorFlow2",
    "tensorflow1",
    "tensorflow2"]

class CounterfactualsImage(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def post(self):
        tf.compat.v1.disable_eager_execution()
        
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

        _id=params["id"]
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
            raise Exception("Either a locally stored model or a URL for the prediction function of the model must be provided.")
                
        
            
        #converting to vector
        try:
            instance=base64_to_vector(instance)
        except Exception as e:  
            return "Could not convert base64 Image to vector: " + str(e)

        #normalizing
        try:
            instance=normalize_img(instance,model_info)
        except Exception as e:
                return  "Could not normalize instance: " + str(e)
        
        if len(model_info["attributes"]["features"]["image"]["shape_raw"])==2 or model_info["attributes"]["features"]["image"]["shape_raw"][-1]==1:
            plt.gray()

        kwargsData = dict(target_proba=None,target_class='other')
        if "target_proba" in params_json:
             kwargsData["target_proba"] = float(params_json["target_proba"])
        if "target_class" in params_json:
             kwargsData["target_class"] = params_json["target_class"]

        size=(6.4, 4.8)
        if "png_height" in params_json and "png_width" in params_json:
            try:
                size=(int(params_json["png_width"])/100.0,int(params_json["png_height"])/100.0)
            except:
                print("Could not convert dimensions for .PNG output file. Using default dimensions.")

        cf = Counterfactual(predic_func, shape=instance.shape, **{k: v for k, v in kwargsData.items() if v is not None})
        explanation = cf.explain(instance)

        pred_class = explanation.cf['class']
        proba = explanation.cf['proba'][0][pred_class]       

        fig, axes = plt.subplots(1,1, figsize = size)
        axes.imshow(explanation.cf['X'].reshape(tuple(model_info["attributes"]["features"]["image"]["shape_raw"])))

        if output_names!=None:
            axes.set_title('Original Class: {}\nCounterfactual Class: {}\nProbability {:.3f}'.format(output_names[explanation.orig_class],output_names[pred_class],proba))  
        else:
            axes.set_title('Original Class: {}\nCounterfactual Class: {}\nProbability {:.3f}'.format(explanation.orig_class,pred_class,proba))  

        #saving
        img_buf = BytesIO()
        fig.savefig(img_buf, bbox_inches='tight')
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response={"type":"image","explanation":b64Image}#,"explanation":json.loads(explanation.to_json())}
        return response

    def get(self):
        return {
        "_method_description": "Finds an image that is similar to the original, but that the model predicts to be from a different class. The class of the conterfactual can be explicitly specified."
                            "This method accepts 5 arguments: " 
                           "the 'id', the 'url' (optional),  the 'params' dictionary (optional) with the configuration parameters of the method, the 'instance' containing the image that will be explained as a matrix, or the 'image' file instead. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally. If provided, then 'url' is ignored.",
        "url": "External URL of the prediction function. This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (images represented as arrays). It must return N outputs (predictions for each image).",
        "instance": "Matrix representing the image to be explained.",
        "image": "Image file to be explained. Ignored if 'instance' was specified in the request. Passing a file is only recommended when the model works with black and white images, or color images that are RGB-encoded using integers ranging from 0 to 255.",
        "params": { 
                "target_class": {
                    "description": "An integer denoting the desired class for the counterfactual instance. Defaults to 'other', a different class from the original.",
                    "type":"int",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "target_proba": {
                    "description": "Float from 0 to 1 representing the target probability for the counterfactual generated. Defaults to 1.0.",
                    "type":"float",
                    "default": 1.0,
                    "range":[0,1],
                    "required":False
                    },
                "png_width":{
                    "description": "Width (in pixels) of the png image containing the explanation.",
                    "type":"int",
                    "default": 640,
                    "range":None,
                    "required":False
                    },
                "png_height": {
                    "description": "Height (in pixels) of the png image containing the explanation.",
                    "type":"int",
                    "default": 480,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "counterfactual_image":"Displays an image that is as similar as possible to the original but that the model predicts to be from a different class."
            },

        "meta":{
                "supportsAPI":True,
                "supportsB&WImage":True,
                "needsData": False,
                "requiresAttributes":[]
            }

        }