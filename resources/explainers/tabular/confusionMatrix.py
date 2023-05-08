from flask_restful import Resource
import joblib
import json
from explainerdashboard import ClassifierExplainer
from explainerdashboard.dashboard_components.classifier_components import ConfusionMatrixComponent
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants


class ConfusionMatrix(Resource):

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
        params_json={}
        if "params" in params:
            params_json=params["params"]

        return self.explain(_id,params_json)


    def explain(self,_id,params_json):
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)


        #getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        target_name=model_info["attributes"]["target_names"][0]
        features=model_info["attributes"]["features"]
        feature_names=list(features.keys())
        output_names=features[target_name]["values_raw"]
        model_task = model_info["model_task"]  

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
            dataframe=dataframe[feature_names]
        else:
            raise Exception("The training data file was not provided.")

        feature_names.remove(target_name)

        #loading model (.pkl file)
        if model_file!=None:
            if backend in ontologyConstants.SKLEARN_URIS:
                model = joblib.load(model_file)
            elif backend in ontologyConstants.XGBOOST_URIS:
                model = joblib.load(model_file)
            elif backend in ontologyConstants.LIGHTGBM_URIS:
                model = joblib.load(model_file)
            else:
                return "This explainer only supports scikit-learn-based models"
        else:
            return "Model file was not uploaded."

        if model_task in ontologyConstants.CLASSIFICATION_URIS:
            explainer = ClassifierExplainer(model, dataframe.drop([target_name], axis=1, inplace=False), dataframe[target_name],labels=output_names)
        else:
            return "AI task not supported. This explainer only supports scikit-learn-based classifiers."

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


    def get(self):
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
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[]
            }
  
        }
    

