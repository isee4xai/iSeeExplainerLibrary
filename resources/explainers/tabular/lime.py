from http.client import BAD_REQUEST
from flask_restful import Resource
from flask import request
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import joblib
import h5py
import json
import lime.lime_tabular
import os
from PIL import Image
from getmodelfiles import get_model_files
import requests
from utils import ontologyConstants
from utils.dataframe_processing import normalize_dataframe
from utils.base64 import PIL_to_base64
from html2image import Html2Image
import traceback

class Lime(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder    

    def post(self):
        try:
            params = request.json
            if params is None:
                return "The json body is missing.",BAD_REQUEST
        
            #Check params
            if("id" not in params):
                return "The model id was not specified in the params.",BAD_REQUEST
            if("type" not in params):
                return "The instance type was not specified in the params.",BAD_REQUEST
            if("instance" not in params):
                return "The instance was not specified in the params.",BAD_REQUEST

            _id =params["id"]
            if("type"  in params):
                inst_type=params["type"]
            instance=params["instance"]
            url=None
            if "url" in params:
                url=params["url"]
            params_json={}
            if "params" in params:
                params_json=params["params"]
        
            #getting model info, data, and file from local repository
            model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

            ## loading data
            if data_file!=None:
                dataframe = joblib.load(data_file) ##error handling?
            else:
                return "The training data file was not provided.",BAD_REQUEST
        

            ##getting params from info
            model_info=json.load(model_info_file)
            backend = model_info["backend"] 
        
            #Checking model task
            model_task = model_info["model_task"]  
            if model_task in ontologyConstants.CLASSIFICATION_URIS:
                model_task="classification"
            elif model_task in ontologyConstants.REGRESSION_URIS:
                model_task="regression"
            else:
                return "Model task not supported: " + model_task

            features=model_info["attributes"]["features"]
            target_name=model_info["attributes"]["target_names"][0]
            feature_names=list(dataframe.columns)
            feature_names.remove(target_name)
            try:
                class_names=features[target_name]["values_raw"]
            except:
                class_names=None
        
            categorical_features=[]
            categorical_names={}
            for feature in feature_names:
                if features[feature]["data_type"]=="categorical":
                    i=dataframe.columns.get_loc(feature)
                    categorical_features.append(i)
                    categorical_names.update({i:[ str(x) for x in features[feature]["values_raw"]]})

            dataframe.drop([target_name], axis=1, inplace=True)

            kwargsData = dict(mode=model_task, feature_names=feature_names, categorical_features=categorical_features,categorical_names=categorical_names, class_names=class_names)

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
                        return "Could not extract prediction function from model: " + str(e),BAD_REQUEST
            elif url!=None:
                def predict(X):
                    return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
                predic_func=predict
            else:
                return "Either a stored model or a valid URL for the prediction function must be provided.",BAD_REQUEST

            #getting params from request

            kwargsData2 = dict(labels=None, top_labels=None, num_features=None)

            if "output_classes" in params_json and params_json["output_classes"] and class_names: #labels (if classification)
                kwargsData2["labels"] = [class_names.index(c) for c in params_json["output_classes"]]
            if "top_classes" in params_json and class_names: #if classification
                kwargsData2["top_labels"] = int(params_json["top_classes"])   #top labels
                if(not kwargsData2["top_labels"]):
                    kwargsData2["top_labels"]=None
            if "num_features" in params_json:
                kwargsData2["num_features"] = int(params_json["num_features"])

            #normalize instance
            df_inst=pd.DataFrame([instance.values()],columns=instance.keys())
            if target_name in df_inst.columns:
                df_inst.drop([target_name], axis=1, inplace=True)
            df_inst=df_inst[feature_names]
            norm_instance=normalize_dataframe(df_inst,model_info).to_numpy()[0]

            explainer = lime.lime_tabular.LimeTabularExplainer(dataframe.to_numpy(),
                                                              **{k: v for k, v in kwargsData.items() if v is not None})
            explanation = explainer.explain_instance(norm_instance, predic_func, **{k: v for k, v in kwargsData2.items() if v is not None}) 

        
            #formatting json explanation
            #ret = explanation.as_map()
            #ret = {str(k):[(int(i),float(j)) for (i,j) in v] for k,v in ret.items()}
            #if kwargsData["class_names"]!=None:
            #    ret = {kwargsData["class_names"][int(k)]:v for k,v in ret.items()}
            #if kwargsData["feature_names"]!=None:
            #    ret = {k:[(kwargsData["feature_names"][i],j) for (i,j) in v] for k,v in ret.items()}
            #ret=json.loads(json.dumps(ret))

            ##saving

            hti = Html2Image()
            hti.output_path= os.getcwd()
            size=(10, 4)
            css="body {background: white;}"
            if "png_height" in params_json and "png_width" in params_json:
                size=(int(params_json["png_width"])/100,int(params_json["png_height"])/100)
                hti.screenshot(html_str=explanation.as_html(),css_str=css, save_as="temp.png", size=size)  
            else:
                hti.screenshot(html_str=explanation.as_html(),css_str=css, save_as="temp.png",size=(1300,350))

            im=Image.open("temp.png")
            b64Image=PIL_to_base64(im)
            os.remove("temp.png")

            response={"type":"image","explanation":b64Image,"explanation_llm":json.loads(json.dumps(dict(explanation.as_list(explanation.available_labels()[0]))))}
            return response

        except:
            return traceback.format_exc(), 500


    def get(self,id=None):
        
        base_dict={
        "_method_description": "LIME perturbs the input data samples in order to train a simple model that approximates the prediction for the given instance and similar ones. "
                           "The explanation contains the weight of each attribute to the prediction value. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url'(optional),  and the 'params' dictionary (optiohnal) with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance not including the target class.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                    "output_classes" : {
                        "description":  "Array of strings representing the classes to be explained. Defaults to class at index 1. This parameter is overriden by 'top_classes' so make sure to set 'top_classes' to 0 to use it",
                        "type":"array",
                        "default": None,
                        "range":None,
                        "required":False
                    },
                    "top_classes":{
                        "description": "Integer representing the number of classes with the highest prediction probability to be explained. Overrides 'output_classes' if provided.",
                        "type":"int",
                        "default": 1,
                        "range":None,
                        "required":False
                    },
                    "num_features": {
                        "description": "Integer representing the maximum number of features to be included in the explanation.",
                        "type":"int",
                        "default": 10,
                        "range":None,
                        "required":False
                    },
                    "png_width":{
                        "description": "Width (in pixels) of the png image containing the explanation.",
                        "type":"int",
                        "default": 1000,
                        "range":None,
                        "required":False
                    },
                    "png_height": {
                        "description": "Height (in pixels) of the png image containing the explanation.",
                        "type":"int",
                        "default": 400,
                        "range":None,
                        "required":False
                    }
            },
        "output_description":{
                "lime_plot": "An image contaning a plot with the most influent features for the given instance. For regression models, the plot displays both positive and negative contributions of each feature value to the predicted outcome."
                "The same applies to classification models, but there can be a plot for each possible class. A table containing the feature values of the instance is also included in the image."
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

            base_dict["params"]["num_features"]["range"]=[1,len(feature_names)]
            base_dict["params"]["num_features"]["default"]=min(10,len(feature_names))

            if model_info["attributes"]["features"][target_name]["data_type"]=="categorical":

                output_names=model_info["attributes"]["features"][target_name]["values_raw"]

                base_dict["params"]["output_classes"]["default"]=[output_names[1]]
                base_dict["params"]["output_classes"]["range"]=output_names

                base_dict["params"]["top_classes"]["range"]=[0,len(output_names)]

                return base_dict

            else:
                base_dict["params"].pop("output_classes")
                base_dict["params"].pop("top_classes")
                return base_dict

        else:
            return base_dict
