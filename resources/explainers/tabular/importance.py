from flask_restful import Resource
import tensorflow as tf
import torch
import joblib
import h5py
import json
import dalex as dx
from flask import request
from PIL import Image
from io import BytesIO
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import PIL_to_base64

class Importance(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder 
        
    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing"
        
        #Check params
        if("id" not in params):
            return "The model id was not specified in the params."

        _id =params["id"]
        if("type"  in params):
            inst_type=params["type"]
        url=None
        if "url" in params:
            url=params["url"]
        params_json={}
        if "params" in params:
            params_json=params["params"]
       
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## loading data
        if data_file!=None:
            dataframe = joblib.load(data_file)
        else:
            raise Exception("The training data file was not provided.")

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  
        target_names=model_info["attributes"]["target_names"]
        feature_names=list(dataframe.columns)
        for target in target_names:
            feature_names.remove(target)
        
        #Checking model task
        model_task = model_info["model_task"]  
        if model_task in ontologyConstants.CLASSIFICATION_URIS:
            model_task="classification"
        elif model_task in ontologyConstants.REGRESSION_URIS:
            model_task="regression"
        else:
            return "Model task not supported: " + model_task

        ## loading model
        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                model=tf.keras.models.load_model(model)
            elif backend in ontologyConstants.SKLEARN_URIS:
                model = joblib.load(model_file)
            elif backend in ontologyConstants.PYTORCH_URIS:
                model = torch.load(model_file)
            else:
                return "The model backend is not supported: " + backend
        else:
            return "Model file was not uploaded."
        


        ## params from the request
        kwargsData = dict()
        if "variables" in params_json and params_json["variables"]:
            kwargsData["variables"] = json.loads(params_json["variables"]) if isinstance(params_json["variables"],str) else params_json["variables"]
       
        explainer = dx.Explainer(model, dataframe.drop(target_names, axis=1, inplace=False), dataframe.loc[:, target_names],model_type=model_task)
        parts = explainer.model_parts(**{k: v for k, v in kwargsData.items()})
        fig=parts.plot(show=False)
        
        #saving
        img_buf = BytesIO()
        fig.write_image(img_buf)
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response={"type":"image","explanation":b64Image}#,"explanation":dict_exp}
        return response


    def get(self,id=None):

        base_dict={
        "_method_description": "This method measures the increase in the prediction error of the model after the feature's values are randomly permuted. " 
                                "A feature is considered important if the error of the model increases significantly when permuting it. Accepts 2 arguments: " 
                            "the 'id' string, and the 'params' object (optional) containing the configuration parameters of the explainer."
                            " These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "variables": {
                    "description": "Array of strings with the names of the features for which the importance will be calculated. Defaults to all features.",
                    "type":"array",
                    "default": None,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "bar_plot": "A bar plot representing the increase in the prediction error (importance) for the features with the highest values."
                },
        "meta":{
                "modelAccess":"File",
                "supportsBWImage":False,
                "needsTrainingData": True
            }
        }

        if id is not None:
            #Getting model info, data, and file from local repository
            try:
                _, model_info_file, data_file = get_model_files(id,self.model_folder)
            except:
                return base_dict


            dataframe = joblib.load(data_file)
            model_info=json.load(model_info_file)
            target_name=model_info["attributes"]["target_names"][0]
            feature_names=list(dataframe.columns)
            feature_names.remove(target_name)

            base_dict["params"]["variables"]["default"]=feature_names
            base_dict["params"]["variables"]["range"]=feature_names

            return base_dict
           
        else:
            return base_dict
         
    