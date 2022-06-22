from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
import shap
from flask import request
import matplotlib.pyplot as plt
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests

class Shap(Resource):
   
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id",required=True)
        parser.add_argument("url")
        parser.add_argument('params',required=True)
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        params_json = json.loads(args.get("params"))
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id)

        ## loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise "The training data file was not provided."

        #getting params from request
        instance = params_json["instance"]
        index=0
        if "output_index" in params_json:
            index=params_json["output_index"];

        ##getting params from info
        model_info=json.loads(json.load(model_info_file))
        backend = model_info["backend"]  ##error handling?

        kwargsData = dict(feature_names=None, output_names=None)
        out_names=None
        if "feature_names" in model_info:
            kwargsData["feature_names"] = model_info["feature_names"]
        if "output_names" in model_info:
            kwargsData["output_names"] = model_info["output_names"]
            try: 
                out_names=kwargsData["output_names"][index]
            except:
                out_names=kwargsData["output_names"]

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
            raise "Either a stored model or a valid URL for the prediction function must be provided."


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
        "_method_description": "This explaining method displays the contribution of each attribute for an individual prediction based on Shapley values. This method accepts 3 arguments: " 
                           "the 'id', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",

        "params": { 
                "instance": "Array with the feature values of an instance without including the target class.",
                "output_index": "(Optional) Integer representing the index of the class to be explained. Ignore for regression models. Default index is 0." 
                },

        "params_example":{
                "instance": [1966, 62, 8, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                "output_index": 2,
               }
  
        }
    

