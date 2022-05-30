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
import lime.lime_text
from html2image import Html2Image
from saveinfo import save_file_info

class LimeText(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()

        model = args.get("model")
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
      
        kwargsData = dict(class_names=None)

        if "class_names" in params_json:
            kwargsData["class_names"] = params_json["class_names"]

        # Create explainer
        explainer = lime.lime_text.LimeTextExplainer(**{k: v for k, v in kwargsData.items() if v is not None})

        kwargsData2 = dict(labels=(1,), top_labels=None, num_features=None)

        if "output_classes" in params_json:
            kwargsData2["labels"] = params_json["output_classes"]  #labels
        if "top_classes" in params_json:
            kwargsData2["top_labels"] = params_json["top_classes"]   #top labels
        if "num_features" in params_json:
            kwargsData2["num_features"] = params_json["num_features"]

        explanation = explainer.explain_instance(instance, predic_func, **{k: v for k, v in kwargsData2.items() if v is not None}) 
        
        #formatting json explanation
        ret = explanation.as_map()
        ret = {str(k):[(int(i),float(j)) for (i,j) in v] for k,v in ret.items()}
        if kwargsData["class_names"]!=None:
            ret = {kwargsData["class_names"][int(k)]:v for k,v in ret.items()}
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
                           "The explanation contains the weight of each word to the prediction value. Requires 2 arguments: " 
                           "the 'params' string and the 'model' which is a file containing the trained model. " 
                           "These arguments are described below.",

        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models. For models with different backends, also upload "
             "a .pkl, and make sure that the prediction function of the model is called 'predict'. This can be achieved by using a wrapper class.",

        "params": { 
                "instance": "A string with the text to be explained.",
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "class_names": "(Optional) Array of strings containing the names of the possible classes.",
                "output_classes" : "(Optional) Array of ints representing the classes to be explained.",
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probablity to be explained.",
                "num_features": "(Optional) Int representing the maximum number of features to be included in the explanation."
                }
        }