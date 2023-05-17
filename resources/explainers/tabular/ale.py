from flask_restful import Resource
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
from alibi.explainers import ALE, plot_ale
import math
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import PIL_to_base64
import requests

class Ale(Resource):
    
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

        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"] 
        target_name=model_info["attributes"]["target_names"][0]
        output_names=model_info["attributes"]["features"][target_name]["values_raw"]
        feature_names=list(dataframe.columns)
        kwargsData = dict(feature_names=feature_names,target_names=output_names)
        feature_names.remove(target_name)

        ## getting predict function
        predic_func=None
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
      
        #getting params from request
        kwargsData2 = dict(features=None)
        if "features_to_show" in params_json and params_json["features_to_show"]:
            features = json.loads(params_json["features_to_show"]) if isinstance(params_json["features_to_show"],str) else params_json["features_to_show"]
            kwargsData2["features"]=[dataframe.columns.get_loc(c) for c in features if c in dataframe]


        proba_ale_lr = ALE(predic_func, **{k: v for k, v in kwargsData.items()})
        proba_exp_lr = proba_ale_lr.explain(dataframe.drop([target_name], axis=1, inplace=False).to_numpy(),**{k: v for k, v in kwargsData2.items()})
        
        if(kwargsData2["features"]!=None):
            dim = math.ceil(len(kwargsData2["features"])**(1/2))
        else:
            dim = math.ceil(len(proba_exp_lr.feature_names)**(1/2))

        fig, ax = plt.subplots(dim, dim, sharey='all');
        plot_ale(proba_exp_lr,ax=ax,fig_kw={'figwidth': 12, 'figheight': 10})
        
        #saving
        img_buf = BytesIO()
        fig.savefig(img_buf,bbox_inches="tight")
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response={"type":"image","explanation":b64Image}#,"explanation":dict_exp}
        return response


    def get(self, id=None):

        base_dict={
            "_method_description": "Computes the accumulated local effects (ALE) of a model for the specified features. This method accepts 3 arguments: " 
                               "the 'id', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                               "These arguments are described below.",

            "id": "Identifier of the ML model that was stored locally.",
            "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
                   "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
            "params": { 
                    "features_to_show": {
                        "description":"Array of strings representing the name of the features to be explained. Defaults to all features.",
                        "type":"array",
                        "default": None,
                        "range":None,
                        "required":False
                        }

                    },
            "output_description":{
                    "ale_plot": "A plot for each of the specified features where the y-axis represents the global feature effect on the outcome value according to the computed ALE values."
                   },

            "meta":{
                    "modelAccess":"Any",
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

            base_dict["params"]["features_to_show"]["default"]=feature_names
            base_dict["params"]["features_to_show"]["range"]=feature_names

            return base_dict
           

        else:
            return base_dict
    
