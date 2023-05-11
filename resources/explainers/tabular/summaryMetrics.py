from flask_restful import Resource
import joblib
import json
from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.dashboard_components.regression_components import RegressionModelSummaryComponent
from explainerdashboard.dashboard_components.classifier_components import ClassifierModelSummaryComponent
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants


class SummaryMetrics(Resource):

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

        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        model_task = model_info["model_task"]

        if model_task not in ontologyConstants.CLASSIFICATION_URIS and model_task not in ontologyConstants.REGRESSION_URIS:
            return "AI task not supported. This explainer only supports scikit-learn-based classifiers or regressors."

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        #loading model (.pkl file)
        if model_file!=None:
            if backend in ontologyConstants.SKLEARN_URIS:
                model = joblib.load(model_file)
            elif backend in ontologyConstants.XGBOOST_URIS:
                model = joblib.load(model_file)
            elif backend in ontologyConstants.LIGHTGBM_URIS:
                model = joblib.load(model_file)
            else:
                return "This explainer only supports scikit-learn-based models."
        else:
            return "Model file was not uploaded."

        return self.explain(model,model_info,dataframe,params_json)


    def explain(self,model,model_info,data,params_json):
        
        #getting params from model info
        target_name=model_info["attributes"]["target_names"][0]
        try:
            output_names=model_info["attributes"]["features"][target_name]["values_raw"]
        except:
            pass
        model_task = model_info["model_task"]

        #getting params from json
        label=None
        if "label" in params_json:
            try:
                label=str(params_json["label"])
            except Exception as e:
                return "Could not convert to label to string: " + str(e)

        if model_task in ontologyConstants.CLASSIFICATION_URIS:
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


    def get(self,id=None):
        return {
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
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[]
            }
  
        }
    

