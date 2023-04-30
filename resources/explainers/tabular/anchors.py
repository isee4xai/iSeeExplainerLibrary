from flask_restful import Resource
from flask import request
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
from alibi.explainers import AnchorTabular
from getmodelfiles import get_model_files
import requests
from utils import ontologyConstants
from utils.dataframe_processing import normalize_dict

class Anchors(Resource):

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
        if("type" not in params):
            return "The instance type was not specified in the params."
        if("instance" not in params):
            return "The instance was not specified in the params."

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


        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"] 
        target_name=model_info["attributes"]["target_names"][0]
        features=model_info["attributes"]["features"]
        feature_names=list(features.keys())
        feature_names.remove(target_name)

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        dataframe.drop([target_name], axis=1, inplace=True)

        categorical_names={}
        for feature in feature_names:
            if features[feature]["data_type"]=="categorical":
                categorical_names.update({dataframe.columns.get_loc(feature):[ str(x) for x in features[feature]["values_raw"]]})

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

        kwargsData = dict(feature_names=feature_names,categorical_names=categorical_names)

        #getting params from request
        kwargsData2 = dict(threshold=0.95)
        if "threshold" in params_json:
            kwargsData2["threshold"] = float(params_json["threshold"])

        #normalize instance
        norm_instance=np.array(list(normalize_dict(instance,model_info).values()))

        # Create data
        explainer = AnchorTabular(predic_func, **{k: v for k, v in kwargsData.items()})

        explainer.fit(dataframe.to_numpy(), disc_perc=(25, 50, 75))
        
        explanation = explainer.explain(norm_instance, **{k: v for k, v in kwargsData2.items()})
        print(explanation.anchor)
        if explanation.anchor:
            ret = dict(anchor=(' AND '.join(explanation.anchor)),precision=explanation.precision, coverage=explanation.coverage)
        else:
            ret = dict(anchor=(' AND '.join(explanation.anchor)),precision=explanation.precision[0], coverage=explanation.coverage)
        return {"type":"dict","explanation":json.loads(json.dumps(ret))}


    def get(self):
        return {
        "_method_description": "Anchors provide local explanations in the form of simple boolean rules with a precision score and a "
                            "coverage value which represents the scope in which that rules applies to similar instances. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url' (optional),  and the 'params' JSON (optional) with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance without including the target class.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",

        "params": { 
                "threshold": {
                    "description": "The minimum level of precision required for the anchors. Default is 0.95",
                    "type":"float",
                    "default": 0.95,
                    "range":[0,1],
                    "required":False
                    }
                },
        "output_description":{
                "anchor_json": "A JSON object with the boolean rule (anchor) that was found, and values for its precision and coverage (scope in which that rules applies to similar instances)."
               },

        "meta":{
                "supportsAPI":True,
                "needsData": True,
                "requiresAttributes":[]
            }
        }
    
