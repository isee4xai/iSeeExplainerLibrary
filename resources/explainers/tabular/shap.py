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
from flask import request
import matplotlib.pyplot as plt
from saveinfo import save_file_info

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
        instance=params_json["instance"]

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
      
        kwargsData = dict(feature_names=None, output_names=None)
        index=0;
        out_names=None
        if "feature_names" in params_json:
            kwargsData["feature_names"] = params_json["feature_names"]
        if "output_index" in params_json:
            index=params_json["output_index"];
        if "output_names" in params_json:
            kwargsData["output_names"] = params_json["output_names"]
            try: 
                out_names=kwargsData["output_names"][index]
            except:
                out_names=kwargsData["output_names"]

        # Create data
        explainer = shap.KernelExplainer(predic_func, dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False),**{k: v for k, v in kwargsData.items()})

        shap_values = explainer.shap_values(np.array(instance))
        
        if(len(np.array(shap_values).shape)!=1):
            explainer.expected_value=explainer.expected_value[index]
            shap_values=shap_values[index]
            
            
        additive_exp = shap.force_plot(explainer.expected_value, shap_values,features=np.array(instance),feature_names=kwargsData["feature_names"],out_names=out_names,show=False)
        ##saving
        upload_folder, filename, getcall = save_file_info(request.path)
        shap.plots._force.save_html(upload_folder+filename+".html",additive_exp)
        shap.force_plot(explainer.expected_value, shap_values,features=np.array(instance), feature_names=kwargsData["feature_names"],out_names=out_names,matplotlib=True,show=False)
        plt.savefig(upload_folder+filename+".png")

        #formatting json output
        shap_values = [x.tolist() for x in shap_values]
        ret=json.loads(json.dumps(shap_values))
        
        #Insert code for image uploading and getting url
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":ret}

  

        return response


    def get(self):
        return {
        "_method_description": "This explaining method displays the contribution of each attribute for an individual prediction based on Shapley values. Requires 3 arguments: " 
                           "the 'params' string, the 'model' which is a file containing the trained model, and " 
                           "the 'data', containing the training data used for the model. These arguments are described below.",

        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models. For models with different backends, also upload "
             "a .pkl, and make sure that the prediction function of the model is called 'predict'. This can be achieved by using a wrapper class.",

        "data": "Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The target class must be the last column of the DataFrame.",
        "params": { 
                "instance": "Array with the feature values of an instance without including the target class.",
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "feature_names": "(Optional) Array of strings corresponding to the columns in the training data. ", #MIGH DELETE IN FUTURE VERSIONS
                "output_names": "(Optional) Array of strings containing the names of the possible classes.",
                "output_index": "(Optional) Integer representing the index of the class to be explained. Ignore for regression models. Default index is 0." 
                },

        "params_example":{
                "instance": [1966, 62, 8, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                "backend": "sklearn",
                "feature_names": ["construction_year", "surface","floor","no_rooms","district_Bemowo",
                                    "district_Bielany","district_Mokotow","district_Ochota","district_Praga",
                                    "district_Srodmiescie","district_Ursus","district_Ursynow",
                                    "district_Wola","district_Zoliborz"],
                "output_names": ["Cheap", "Average", "Expensive"],
               }
  
        }
    

