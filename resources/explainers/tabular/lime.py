from flask_restful import Resource,reqparse
from flask import request
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import joblib
import werkzeug
import h5py
import json
import lime.lime_tabular
from html2image import Html2Image
from saveinfo import save_file_info

class Lime(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("data", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        
        data = args.get("data")
        model = args.get("model")
        dataframe = joblib.load(data)
        params_json = json.loads(args.get("params"))
        instance=params_json["instance"]
        backend = params_json["backend"]
       
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
      
        kwargsData = dict(mode="classification",training_labels=None, feature_names=None, categorical_features=None,categorical_names=None, class_names=None)

        
        if "model_task" in params_json:
            kwargsData["mode"] = params_json["model_task"]
        if "training_labels" in params_json:
            kwargsData["training_labels"] = params_json["training_labels"]
        if "feature_names" in params_json:
            kwargsData["feature_names"] = params_json["feature_names"]
        if "categorical_features" in params_json:
            kwargsData["categorical_features"] = params_json["categorical_features"]
        if "categories_names" in params_json:
            kwargsData["categorical_names"] = {int(k):v for k,v in params_json["categories_names"].items()}
        if "class_names" in params_json:
            kwargsData["class_names"] = params_json["class_names"]

        # Create data
        explainer = lime.lime_tabular.LimeTabularExplainer(dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False).to_numpy(),
                                                            **{k: v for k, v in kwargsData.items() if v is not None})

        kwargsData2 = dict(labels=(1,), top_labels=None, num_features=None)

        if "output_classes" in params_json:
            kwargsData2["labels"] = params_json["output_classes"]  #labels
        if "top_classes" in params_json:
            kwargsData2["top_labels"] = params_json["top_classes"]   #top labels
        if "num_features" in params_json:
            kwargsData2["num_features"] = params_json["num_features"]

        explanation = explainer.explain_instance(np.array(instance, dtype='f'), predic_func, **{k: v for k, v in kwargsData2.items() if v is not None}) 
        
        #formatting json explanation
        ret = explanation.as_map()
        ret = {str(k):[(int(i),float(j)) for (i,j) in v] for k,v in ret.items()}
        if kwargsData["class_names"]!=None:
            ret = {kwargsData["class_names"][int(k)]:v for k,v in ret.items()}
        if kwargsData["feature_names"]!=None:
            ret = {k:[(kwargsData["feature_names"][i],j) for (i,j) in v] for k,v in ret.items()}
        ret=json.loads(json.dumps(ret))

        ##saving
        upload_folder, filename, getcall = save_file_info(request.path)
        hti = Html2Image()
        hti.output_path= upload_folder
        hti.screenshot(html_str=explanation.as_html(), save_as=filename+".png")   
        explanation.save_to_file(upload_folder+filename+".html")
        
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":ret}
        return response

    def get(self):
        return {
        "_method_description": "LIME perturbs the input data samples in order to train a simple model that approximates the prediction for the given instance and similar ones. "
                           "The explanation contains the weight of each attribute to the prediction value. Requires 3 arguments: " 
                           "the 'params' string, the 'model' which is a file containing the trained model, and " 
                           "the 'data', containing the training data used for the model. These arguments are described below.",

        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models. For models with different backends, also upload "
             "a .pkl, and make sure that the prediction function of the model is called 'predict'. This can be achieved by using a wrapper class.",

        "data": "Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The target class must be the last column of the DataFrame",

        "params": { 
                "instance": "Array representing a row with the feature values of an instance not including the target class.",
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "model_task": "(Optional) A string containing 'classification' or 'regression' accordingly. Defaults to 'classification'.",
                "training_labels": "(Optional) Array of ints representing labels for training data.",
                "feature_names": "(Optional) Array of strings corresponding to the columns in the training data. ", #MIGH DELETE IN FUTURE VERSIONS
                "categorical_features": "(Optional) Array of ints representing the indexes of the categorical columns. Columns not included here will be considered continuous.",
                "categories_names": "(Optional) Dictionary which int keys representing the indexes of the categorical columns, each key having as value an array of strings" 
                "with the names of the different categories for that feature.",
                "class_names": "(Optional) Array of strings containing the names of the possible classes.",
                "output_classes" : "(Optional) Array of ints representing the classes to be explained.",
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probability to be explained.",
                "num_features": "(Optional) Int representing the maximum number of features to be included in the explanation."
                },

        "params_example":{
                "backend": "sklearn",
                "instance": [1966, 62, 8, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                "feature_names": ["construction_year", "surface","floor","no_rooms","district_Bemowo",
                                    "district_Bielany","district_Mokotow","district_Ochota","district_Praga",
                                    "district_Srodmiescie","district_Ursus","district_Ursus","district_Ursynow",
                                    "district_Wola","district_Zoliborz"],
                "categorical_features": [4,5,6,7,8,9,10,11,12,13,14],
                "class_names": ["Cheap", "Expensive"],
                "num_features": 6,
    
               }

        }