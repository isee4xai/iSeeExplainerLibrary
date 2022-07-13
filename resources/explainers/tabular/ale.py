from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
from alibi.explainers import ALE, plot_ale
import math
import matplotlib.pyplot as plt
from flask import request
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests

class Ale(Resource):
    
    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id",required=True)
        parser.add_argument("url")
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        params_json = json.loads(args.get("params"))
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?

        kwargsData = dict(feature_names=None,target_names=None)
        if "feature_names" in model_info:
            kwargsData["feature_names"]=model_info["feature_names"]
        if "output_names" in model_info:
            kwargsData["target_names"] = model_info["output_names"]

        ## getting predict function
        predic_func=None
        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend=="sklearn":
                mlp = joblib.load(model_file)
                predic_func=mlp.predict_proba
            elif backend=="PYT":
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
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")
      
        #getting params from request
        kwargsData2 = dict(features=None)
        if "features_to_show" in params_json:
            kwargsData2["features"] = params_json["features_to_show"]


        proba_ale_lr = ALE(predic_func, **{k: v for k, v in kwargsData.items()})
        proba_exp_lr = proba_ale_lr.explain(dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False).to_numpy(),**{k: v for k, v in kwargsData2.items()})
        
        
        if(kwargsData2["features"]!=None):
            dim = math.ceil(len(kwargsData2["features"])**(1/2))
        else:
            dim = math.ceil(len(proba_exp_lr.feature_names)**(1/2))

        fig, ax = plt.subplots(dim, dim, sharey='all');
        plot_ale(proba_exp_lr,ax=ax,fig_kw={'figwidth': 12, 'figheight': 10})
        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        fig.savefig(upload_folder+filename+'.png')

        response = {"plot_png":getcall+'.png',"explanation":json.loads(proba_exp_lr.to_json())}
        return response


    def get(self):
        return {
        "_method_description": "Computes the accumulated local effects (ALE) of a model for the specified features. The outcome represents the global feature effect on the prediction probability. This method accepts 3 arguments: " 
                           "the 'id', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "features_to_show": "(Optional) Array of ints representing the indices of the features to be explained. Defaults to all features"
                }
        }
    