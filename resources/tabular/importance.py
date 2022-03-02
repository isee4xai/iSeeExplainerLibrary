from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import joblib
import werkzeug
import h5py
import json
import dalex as dx

class Importance(Resource):
   
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
          
         
        elif backend=="sklearn":
            mlp = joblib.load(model)
         
        else:
            mlp = torch.load(model)
          
        kwargsData = dict()
            
        if "variables" in params_json:
            kwargsData["variables"] = params_json["variables"]

        explainer = dx.Explainer(mlp, dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False), dataframe.iloc[:,-1:],model_type="classification")
        parts = explainer.model_parts(**{k: v for k, v in kwargsData.items()})
        
        response={"plot_url":"","explanation":json.loads(parts.result.to_json())}
        return response


    def get(self):
        return {
        "_method_description": "This method measures the increase in the prediction error of the model after the feature's values are randomly permuted. " 
                                "A feature is considered important if the error of the model increases significantly when permuting it. Requires 3 arguments: " 
                           "the 'params' string, the 'model' which is a file containing the trained model, and " 
                           "the 'data', containing the training data used for the model. These arguments are described below.",
        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models.",

        "data": "Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The target class must be the last column of the DataFrame.",

        "params": { 
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "variables": "(Optional) Array of strings with the names of the features for which the importance will be calculated."
                
                },

        "params_example":{
                "backend": "sklearn",
                "variables": [
                    "1. Most of the time I have difficulty concentrating on simple tasks",
                    "2. I don't feel like doing my daily duties",
                    "3. My friends or family have told me that I look different",
                    "4. When I think about the future it is difficult for me to imagine it clearly",
                    "5. People around me often ask me how I feel",
                    "6. I consider that my life is full of good things",
                    "7. My hobbies are still important to me"
                  ]
               }
        }
    