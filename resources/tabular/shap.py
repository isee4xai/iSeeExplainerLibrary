from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import joblib
import werkzeug
import h5py
import json
import shap

class Shap(Resource):
   
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("data", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        
        data = args.get("data")
        dataframe = joblib.load(data)
        params_json = json.loads(args.get("params"))
        instances=params_json["instances"]

        backend = params_json["backend"]
       
        model = args.get("model")
        
        if backend=="TF1" or backend=="TF2":
            model=h5py.File(model, 'w')
            mlp = tf.keras.models.load_model(model)
            predic_func=mlp.predict
            mlp = tf.keras.models.load_model(model)
        elif backend=="sklearn":
            mlp = joblib.load(model)
            predic_func =mlp.predict_proba
        else:
            mlp = torch.load(model)
            predic_func=mlp.predict
      
        kwargsData = dict(feature_names=None, output_names=None)

        if "feature_names" in params_json:
            kwargsData["feature_names"] = params_json["feature_names"]
        if "output_names" in params_json:
            kwargsData["output_names"] = params_json["output_names"]

        # Create data
        explainer = shap.KernelExplainer(predic_func, dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False),**{k: v for k, v in kwargsData.items()})

        shap_values = explainer.shap_values(np.array(instances))

        shap_values = [x.tolist() for x in shap_values]
        
        ret=json.loads(json.dumps(shap_values))
        
        #Insert code for image uploading and getting url
        response={"plot_url":"","explanation":ret}

        return response


    def get(self):
        return {
        "_method_description": "This explaining method displays the contribution of each attribute for an individual prediction based on Shapley values. Requires 3 arguments: " 
                           "the 'params' string, the 'model' which is a file containing the trained model, and " 
                           "the 'data', containing the training data used for the model. These arguments are described below.",
        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
        "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models.",

        "data": "Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The target class must be the last column of the DataFrame.",
        "params": { 
                "instances": "Array of arrays, where each one represents a row with the feature values of an instance without including the target class.",
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "feature_names": "(Optional) Array of strings corresponding to the columns in the training data. ", #MIGH DELETE IN FUTURE VERSIONS
                "output_names": "(Optional) Array of strings containing the names of the possible classes.",
                },

        "params_example":{
                "instances": [ [1966, 62, 8, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [2001, 42, 3, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
                "backend": "sklearn",
                "feature_names": ["construction_year", "surface","floor","no_rooms","district_Bemowo",
                                    "district_Bielany","district_Mokotow","district_Ochota","district_Praga",
                                    "district_Srodmiescie","district_Ursus","district_Ursus","district_Ursynow",
                                    "district_Wola","district_Zoliborz"],
                "output_names": ["Cheap", "Expensive"],
               }
  
        }
    

