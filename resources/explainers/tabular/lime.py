from flask_restful import Resource,reqparse
from flask import request
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
import lime.lime_tabular
from html2image import Html2Image
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests

class Lime(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder    

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id',required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument('url')
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
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
        kwargsData = dict(mode="classification", feature_names=None, categorical_features=None,categorical_names=None, class_names=None)
        if "model_task" in model_info:
            kwargsData["mode"] = model_info["model_task"]
        if "feature_names" in model_info:
            kwargsData["feature_names"] = model_info["feature_names"]
        if "categorical_features" in model_info:
            kwargsData["categorical_features"] = model_info["categorical_features"]
        if "categorical_names" in model_info:
            kwargsData["categorical_names"] = {int(k):v for k,v in model_info["categorical_names"].items()}
        if "output_names" in model_info:
            kwargsData["class_names"] = model_info["output_names"]

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
        kwargsData2 = dict(labels=(1,), top_labels=None, num_features=None)
        if "output_classes" in params_json:
            kwargsData2["labels"] = params_json["output_classes"]  #labels
        if "top_classes" in params_json:
            kwargsData2["top_labels"] = params_json["top_classes"]   #top labels
        if "num_features" in params_json:
            kwargsData2["num_features"] = params_json["num_features"]



        explainer = lime.lime_tabular.LimeTabularExplainer(dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False).to_numpy(),
                                                            **{k: v for k, v in kwargsData.items() if v is not None})
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
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        hti = Html2Image()
        hti.output_path= upload_folder
        hti.screenshot(html_str=explanation.as_html(), save_as=filename+".png")   
        explanation.save_to_file(upload_folder+filename+".html")
        
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":ret}
        return response

    def get(self):
        return {
        "_method_description": "LIME perturbs the input data samples in order to train a simple model that approximates the prediction for the given instance and similar ones. "
                           "The explanation contains the weight of each attribute to the prediction value. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url'(optional),  and the 'params' dictionary (optiohnal) with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance not including the target class.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "output_classes" : "(Optional) Array of ints representing the classes to be explained.",
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probability to be explained. Overrides 'output_classes' if provided.",
                "num_features": "(Optional) Int representing the maximum number of features to be included in the explanation."
                }

        }