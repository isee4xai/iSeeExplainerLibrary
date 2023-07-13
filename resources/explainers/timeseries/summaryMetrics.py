from http.client import BAD_REQUEST
from flask_restful import Resource
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import requests
import h5py
from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.dashboard_components.regression_components import RegressionModelSummaryComponent
from explainerdashboard.dashboard_components.classifier_components import ClassifierModelSummaryComponent
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants
import traceback

class TSSummaryMetrics(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing.",BAD_REQUEST
        
        #Check params
        if("id" not in params):
            return "The model id was not specified in the params.",BAD_REQUEST

        _id =params["id"]
        url=None
        if "url" in params:
            url=params["url"]
        params_json={}
        if "params" in params:
            params_json=params["params"]

        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        model_task = model_info["model_task"]

        if model_task not in ontologyConstants.CLASSIFICATION_URIS and model_task not in ontologyConstants.REGRESSION_URIS:
            return "AI task not supported. This explainer only supports scikit-learn-based classifiers or regressors.",BAD_REQUEST

        #loading data
        if data_file!=None:
            dataframe = pd.read_csv(data_file,header=0)
        else:
            return "The training data file was not provided.",BAD_REQUEST

        ## getting predict function
        predic_func=None
        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp.predict
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

        return self.explain(predic_func,model_info,dataframe,params_json)


    def explain(self,predic_func,model_info,data,params_json):
        
        try:
            #getting params from model info
            target_name=model_info["attributes"]["target_names"][0]
            try:
                output_names=model_info["attributes"]["features"][target_name]["values_raw"]
                n_classes=len(output_names)
            except:
                pass
            model_task = model_info["model_task"]

            #getting params from json
            label=None
            if "label" in params_json:
                try:
                    label=str(params_json["label"])
                except Exception as e:
                    return "Could not convert to label to string: " + str(e),BAD_REQUEST

            class ModelWrapper:
                def predict(self,x):
                    return np.array(predic_func(x))

            if model_task in ontologyConstants.CLASSIFICATION_URIS:

                model=ModelWrapper()
                pred=np.array(predic_func(data.drop([target_name],axis=1).iloc[0:1].to_numpy()))
            
                if(len(pred.shape)!=2): # prediction function only returns class
                    def predict_proba(input):
                        np.array([[0 if i!=x else 1 for i in range(n_classes)] for x in np.array(predic_func(input))],dtype='float32')
                    model.predict_proba=predict_proba
                else:
                    model.predict_proba=predic_func

                explainer = ClassifierExplainer(model, data.drop([target_name], axis=1, inplace=False), data[target_name],labels=output_names,target=target_name)
                if label is None:
                    label=output_names[explainer.pos_label]
                exp=ClassifierModelSummaryComponent(explainer,title="Model performance metrics for Class " + str(label),pos_label=label)
            elif model_task in ontologyConstants.REGRESSION_URIS:
                explainer = RegressionExplainer(model, data.drop([target_name], axis=1, inplace=False), data[target_name],target=target_name)
                exp=RegressionModelSummaryComponent(explainer)

            exp_html=exp.to_html().replace('\n', ' ').replace("\"","'")

            response={"type":"html","explanation":exp_html}
            return response
        except:
            return traceback.format_exc(), 500

    def get(self,id=None):
        
        base_dict={
        "_method_description": "Displays a summary of the performance metrics of the model based on the training dataset. Only supports scikit-learn-based models. This method accepts 2 arguments: " 
                           "the model 'id' and the 'params' object.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "label":{
                    "description": "String with the name of the label that will be considered the positive class. Only for used for classifier models. Defaults to class at index 1 in configuration file.",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "metrics_table": "Displays a summary of the performance metrics of the model."
         },
        "meta":{
                "modelAccess":"File",
                "supportsBWImage":False,
                "needsTrainingData": True
            }
  
        }

        if id is not None:
            #Getting model info, data, and file from local repository
            try:
                _, model_info_file, _ = get_model_files(id,self.model_folder)
                model_info=json.load(model_info_file)
            except:
                return base_dict

            target_name=model_info["attributes"]["target_names"][0]
            output_names=model_info["attributes"]["features"][target_name]["values_raw"]

            base_dict["params"]["label"]["range"]=output_names
            base_dict["params"]["label"]["default"]=output_names[1]

            return base_dict

        else:
            return base_dict
    

