from flask_restful import Resource
import joblib
import json
import tensorflow as tf
import torch
import requests
import h5py
import numpy as np
import pandas as pd
from explainerdashboard import ClassifierExplainer
from explainerdashboard.dashboard_components.classifier_components import ConfusionMatrixComponent
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants


class TSConfusionMatrix(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing."
        
        #Check params
        if("id" not in params):
            return "The model id was not specified in the params."

        _id =params["id"]
        url=None
        if "url" in params:
            url=params["url"]
        params_json={}
        if "params" in params:
            params_json=params["params"]

        return self.explain(_id,params_json,url)


    def explain(self,_id,params_json,url):
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        #loading data
        if data_file!=None:
            dataframe = pd.read_csv(data_file,header=0)
        else:
            raise Exception("The training data file was not provided.")

        #getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        target_name=model_info["attributes"]["target_names"][0]
        features=model_info["attributes"]["features"]
        output_names=features[target_name]["values_raw"]
        model_task = model_info["model_task"]  
        n_classes=len(output_names)

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
                    return "Could not extract prediction function from model: " + str(e)
        elif url!=None:
            def predict(X):
                return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
            predic_func=predict
        else:
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")

        #check univariate
        if(1):
            pass
        else:
            return "This method only supports univariate timeseries datasets."

        if model_task not in ontologyConstants.CLASSIFICATION_URIS:
            return "AI task not supported. This explainer only supports classifiers."

        class ModelWrapper:
            def predict_proba(self,x):
                return np.array(predic_func(x))

        model=ModelWrapper()

        pred=np.array(predic_func(dataframe.drop([target_name],axis=1).iloc[0:1].to_numpy()))

        if(len(pred.shape)!=2): # prediction function only returns class
            def predict_proba(input):
                np.array([[0 if i!=x else 1 for i in range(n_classes)] for x in np.array(predic_func(input))],dtype='float32')
            model.predict_proba=predict_proba

        explainer = ClassifierExplainer(model, dataframe.drop([target_name], axis=1, inplace=False), dataframe[target_name],labels=output_names)
            

        #getting params from request
        cutoff=0.5
        if "cutoff" in params_json:
            try:
                cutoff=float(params_json["cutoff"])
            except Exception as e:
                return "Could not convert to cuttoff to float: " + str(e)


        exp=ConfusionMatrixComponent(explainer,cutoff=cutoff,binary=False)
        exp_html=exp.to_html().replace('\n', ' ').replace("\"","'")

        response={"type":"html","explanation":exp_html}
        return response


    def get(self,id=None):
        return {
        "_method_description": "Displays the confusion matrix of the model using the training dataset. Only supports scikit-learn-based models. This method accepts 2 arguments: " 
                           "the model 'id' and the 'params' object.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "cutoff":{
                    "description": "Float value for the cutoff to consider when building the confusion matrix.",
                    "type":"float",
                    "default": 0.5,
                    "range":[0,1],
                    "required":False
                    } 
                },
        "output_description":{
                "confusion_matrix": "Each row of the matrix represents the instances in an actual class while each column represents the instances in a predicted class."
         },
        "meta":{
                "modelAccess":"File",
                "supportsBWImage":False,
                "needsTrainingData": True
        }
  
    }
    

