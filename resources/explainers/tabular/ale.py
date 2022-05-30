from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import joblib
import werkzeug
import h5py
import json
from alibi.explainers import ALE, plot_ale
import math
import matplotlib.pyplot as plt
from saveinfo import save_file_info
from flask import request


class Ale(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("data", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        
        data = args.get("data")
        dataframe = joblib.load(data)
        params_json = json.loads(args.get("params"))
       

        backend = params_json["backend"]
        
       
        model = args.get("model")
        
        if backend=="TF1" or backend=="TF2":
            model=h5py.File(model, 'w')
            mlp = tf.keras.models.load_model(model)
            predic_func=mlp.predict
        elif backend=="sklearn":
            mlp = joblib.load(model)
            if hasattr(mlp,'predict_proba'):
                predic_func=mlp.predict_proba
            else:
                predic_func=mlp.predict
        elif backend=="PYT":
            mlp = torch.load(model)
            predic_func=mlp.predict
        else:
            mlp = joblib.load(model)
            predic_func=mlp.predict
      
        kwargsData = dict(feature_names = None, target_names=None)
        kwargsData2 = dict(features=None)
            
        if "feature_names" in params_json:
            kwargsData["feature_names"] = params_json["feature_names"]
        if "target_names" in params_json:
            kwargsData["target_names"] = params_json["target_names"]
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
        upload_folder, filename, getcall = save_file_info(request.path)
        fig.savefig(upload_folder+filename+'.png')

        response = {"plot_png":getcall+'.png',"explanation":json.loads(proba_exp_lr.to_json())}
        return response


    def get(self):
        return {
        "_method_description": "Computes the accumulated local effects (ALE) of a model for the specified features. The outcome represents the global feature effect on the prediction probability. Requires 3 arguments: " 
                           "the 'params' string, the 'model' which is a file containing the trained model, and " 
                           "the 'data', containing the training data used for the model. These arguments are described below.",
        
        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models. For models with different backends, also upload "
             "a .pkl, and make sure that the prediction function of the model is called 'predict'. This can be achieved by using a wrapper class.",

        "data": "Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The target class must be the last column of the DataFrame.",

        "params": { 
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "feature_names": "(Optional) Array of strings corresponding to the columns in the training data. ", #MIGH DELETE IN FUTURE VERSIONS
                "target_names":  "(Optional) Array of strings containing the names of the possible classes.",
                "features_to_show": "(Optional) Array of ints representing the indices of the features to be explained."
                },

        "params_example":{
                    "backend": "sklearn",
                    "feature_names": [
                    "construction_year",
                    "surface",
                    "floor",
                    "no_rooms",
                    "district_Bemowo",
                    "district_Bielany",
                    "district_Mokotow",
                    "district_Ochota",
                    "district_Praga",
                    "district_Srodmiescie",
                    "district_Ursus",
                    "district_Ursynow",
                    "district_Wola",
                    "district_Zoliborz"
                  ],
                  "target_names": [ "Cheap", "Expensive" ],
                  "features_to_show": [ 0, 1, 2, 3 ]
  
                  }
       
        }
    