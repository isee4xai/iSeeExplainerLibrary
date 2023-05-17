from flask_restful import Resource
from flask import request
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
import lime.lime_text
import os
from html2image import Html2Image
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import PIL_to_base64
from PIL import Image
import requests

class LimeText(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing."
        
        #Check params
        if("id" not in params):
            return "The model id was not specified in the params."
        if("type" not in params):
            return "The instance type was not specified in the params."
        if("instance" not in params):
            return "The instance was not specified in the params."

        _id =params["id"]
        if("type"  in params):
            inst_type=params["type"]
        instance=params["instance"]
        url=None
        if "url" in params:
            url=params["url"]
        params_json={}
        if "params" in params:
            params_json=params["params"]
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"] 

        label=model_info["attributes"]["target_names"][0]
        
        try:
            output_names=model_info["attributes"]["features"][label]["values_raw"]
        except:
            output_names=None

        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend in ontologyConstants.SKLEARN_URIS:
                mlp = joblib.load(model_file)
                try:
                    predic_func=mlp.predict_proba
                except:
                    predic_func=mlp.predict
            elif backend in ontologyConstants.PYTORCH_URIS:
                mlp = torch.load(model_file)
                predic_func=mlp.predict
            else:
                try:
                    mlp = joblib.load(model_file)
                    predic_func=mlp.predict
                except Exception as e:
                    return "Could not extract prediction function from model: " + str(e)
        elif url!=None:
            def predict(X):
                return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
            predic_func=predict
        else:
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")
        
     
        # Create explainer
        explainer = lime.lime_text.LimeTextExplainer(class_names=output_names)
        kwargsData2 = dict(labels=None, top_labels=1, num_features=None)
        if "output_classes" in params_json:
            kwargsData2["labels"] = json.loads(params_json["output_classes"]) if isinstance(params_json["output_classes"],str) else params_json["output_classes"]  #labels
        if "top_classes" in params_json:
            kwargsData2["top_labels"] = int(params_json["top_classes"])   #top labels
        if "num_features" in params_json:
            kwargsData2["num_features"] = int(params_json["num_features"])

        explanation = explainer.explain_instance(instance, predic_func, **{k: v for k, v in kwargsData2.items() if v is not None}) 
        
        ##formatting json explanation
        #ret = explanation.as_map()
        #ret = {str(k):[(int(i),float(j)) for (i,j) in v] for k,v in ret.items()}
        #if output_names!=None:
        #    ret = {output_names[int(k)]:v for k,v in ret.items()}
        #ret=json.loads(json.dumps(ret))

        #saving
        hti = Html2Image()
        hti.output_path= os.getcwd()
        size=(10, 4)
        css="body {background: white;}"
        if "png_height" in params_json and "png_width" in params_json:
            size=(int(params_json["png_width"])/100,int(params_json["png_height"])/100)
            hti.screenshot(html_str=explanation.as_html(), css_str=css, save_as="temp.png", size=size)   
        else:
            hti.screenshot(html_str=explanation.as_html(),css_str=css, save_as="temp.png")

        im=Image.open("temp.png")
        b64Image=PIL_to_base64(im)
        os.remove("temp.png")

        response={"type":"image","explanation":b64Image}
        return response

    def get(self,id=None):
        return {
        "_method_description": "LIME perturbs the input data samples in order to train a simple model that approximates the prediction for the given instance and similar ones. "
                           "The explanation contains the weight of each word to the prediction value. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "A string with the text to be explained.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "output_classes" : {
                    "description":  "Array of integers representing the classes to be explained.",
                    "type":"array",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "top_classes":{
                        "description": "Integer representing the number of classes with the highest prediction probability to be explained. Overrides 'output_classes' if provided.",
                        "type":"int",
                        "default": 1,
                        "range":None,
                        "required":False
                    },
                "num_features": {
                        "description": "Integer representing the maximum number of features to be included in the explanation.",
                        "type":"int",
                        "default": 10,
                        "range":None,
                        "required":False
                    },
                "png_width":{
                    "description": "Width (in pixels) of the png image containing the explanation.",
                    "type":"int",
                    "default": 1000,
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
                "lime_plot": "An image contaning a plot with the most influyent words for the given instance. For regression models, the plot displays both positive and negative contributions of each word to the predicted outcome."
                "The same applies to classification models, but there can be a plot for each possible class. The text instance with highlighted words is included in the explanation."
               },
        "meta":{
                "modelAccess":"Any",
                "supportsBWImage":False,
                "needsTrainingData": False
         }
        }
