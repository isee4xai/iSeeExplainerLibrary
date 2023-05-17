from flask_restful import Resource
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
import requests
from alibi.explainers import ALE
import matplotlib.pyplot as plt
import seaborn as sns
from flask import request
from getmodelfiles import get_model_files
from io import BytesIO
from PIL import Image
from utils import ontologyConstants
from utils.base64 import PIL_to_base64


class IREX(Resource):
    
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
        target_name=model_info["attributes"]["target_names"][0]
        try:
            output_names=model_info["attributes"]["features"][target_name]["values_raw"]
        except:
            output_names=None
        feature_names=list(dataframe.columns)
        feature_names.remove(target_name)
        kwargsData = dict(feature_names=feature_names,target_names=output_names)

        dataframe.drop([target_name], axis=1, inplace=True)

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
                    return "Could not extract prediction function from model: " + str(e)
        elif url!=None:
            def predict(X):
                return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
            predic_func=predict
        else:
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")
      
        #getting params from request
        threshold=0.01
        classes_to_show=None
        if "expected_answers" in params_json:  
            expected=json.loads(params_json["expected_answers"]) if isinstance(params_json["expected_answers"],str) else params_json["expected_answers"]
        else:
            return "This method requires the expected answers parameter."
        if "threshold" in params_json:
            threshold = float(params_json["threshold"])
        if "classes_to_show" in params_json and params_json["classes_to_show"]:
            classes_to_show = [output_names.index(c) for c in params_json["classes_to_show"]]


        proba_ale_lr = ALE(predic_func, **{k: v for k, v in kwargsData.items()})
        proba_exp_lr = proba_ale_lr.explain(dataframe.to_numpy())

        anomalies=[]
        for i in range(len(proba_exp_lr.ale_values)):
            if(proba_exp_lr.ale_values[i][expected[i]][-1]<0-threshold):
                anomalies.append(i)

        arranged_ale=[]
        if classes_to_show is None:
            classes_to_show=range(proba_exp_lr.ale_values[0].shape[-1])
        i=0
        for oclass in classes_to_show:
            arranged_ale.append([])
            for q in range(len(proba_exp_lr.ale_values)):
                arranged_ale[i].append(abs(proba_exp_lr.ale_values[q][expected[q]][oclass]))
            i=i+1

        ale_df=pd.DataFrame(arranged_ale)
        if kwargsData["target_names"] is None:
            class_names=["Class " +str(i) for i in classes_to_show]
        else:
            class_names=[kwargsData["target_names"][oclass] for oclass in classes_to_show]
        ale_df.index=class_names
        if kwargsData["feature_names"] is None:
            kwargsData["feature_names"]=range(1,len(proba_exp_lr.ale_values)+1)
        ale_df.columns=kwargsData["feature_names"]
        ale_df=ale_df.transpose()

        df_mask = [[False for _ in range(len(proba_exp_lr.ale_values))]]
        for ind in anomalies:
            df_mask[0][ind] = True
        df_mask = pd.DataFrame(df_mask)
        df_masks=[df_mask for _ in classes_to_show]
        mask=pd.concat(df_masks, ignore_index = True, axis = 0)
        plt.figure(figsize = (len(proba_exp_lr.ale_values)/2,len(classes_to_show)*3))
        sns.heatmap(ale_df.transpose(), cmap="Reds", cbar=False)
        sns.heatmap(ale_df.transpose(), cmap="Blues",yticklabels=True, xticklabels=True,mask=mask.to_numpy())

        #saving
        img_buf = BytesIO()
        plt.savefig(img_buf,bbox_inches="tight")
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response = {"type":"image","explanation":b64Image}#"explanation":json.loads(proba_exp_lr.to_json())}
        return response

    def get(self,id=None):
        
        base_dict={
        "_method_description": "IREX is a reusable method for the Iterative Refinement and EXplanation of classification models. It has been designed for domain-expert users -without machine learning skills- that need to understand" 
        " and improve classification models. This particular implementation of IREX uses ALE to identify anomalous features that may be contradictory to what the expert knowledge indicates. Anomalous features are highlighted in red in an ALE heatmap. This method accepts 3 arguments: " 
                           "the 'id', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "expected_answers":{
                    "description": "Array containing the expected answers (according to experts) to the questions of a questionnaire that supossedly contribute to the target class.",
                    "type":"array",
                    "default": None,
                    "range":None,
                    "required":True
                    },
                
                "threshold": {
                    "description": "A float between 0 and 1 for the threshold that will be used to determine anomalous variables. If a feature seems to be contradictory but its absolute ALE value is below this threshold, it will not be considered anomalous. Defaults to 0.01.",
                    "type":"float",
                    "default": 0.01,
                    "range":[0,1],
                    "required":False
                    },
                "classes_to_show": {
                    "description":"Array of string representing the names of the classes to be explained. Defaults to all classes.",
                    "type":"array",
                    "default": None,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "heatmap": "A heatmap displaying the relevance of the features according to ALE, where anomalous features (behavior differring from expected values) are highlighted in red."
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
                _, model_info_file, _ = get_model_files(id,self.model_folder)
            except:
                return base_dict

            model_info=json.load(model_info_file)
            target_name=model_info["attributes"]["target_names"][0]
            output_names=model_info["attributes"]["features"][target_name]["values_raw"]

            base_dict["params"]["classes_to_show"]["default"]=output_names
            base_dict["params"]["classes_to_show"]["range"]=output_names
        
            return base_dict
        else:
            return base_dict
    
