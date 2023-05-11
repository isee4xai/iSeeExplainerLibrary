from flask_restful import Resource
import joblib
import json
from explainerdashboard import RegressionExplainer
from explainerdashboard.dashboard_components.regression_components import PredictedVsActualComponent
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants


class RegressionPredictedVsActual(Resource):

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

        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        model_task = model_info["model_task"]

        if model_task not in ontologyConstants.REGRESSION_URIS:
            return "AI task not supported. This explainer only supports scikit-learn-based regressors."

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

        return self.explain(model,model_info,dataframe)


    def explain(self,model,model_info,data):
        
        #getting params from model info
        target_name=model_info["attributes"]["target_names"][0]

        explainer = RegressionExplainer(model, data.drop([target_name], axis=1, inplace=False), data[target_name],target=target_name)
        exp=PredictedVsActualComponent(explainer)
        exp_html=exp.to_html().replace('\n', ' ').replace("\"","'")

        response={"type":"html","explanation":exp_html}
        return response


    def get(self):
        return {
        "_method_description": "Displays a plot of values predicted for the instances against their actual target values.. Only supports scikit-learn-based models. This method accepts only 1 argument: " 
                           "the model 'id'.",
        "id": "Identifier of the ML model that was stored locally.",
        "output_description":{
                "predicted_vs_actual": "Shows the values predicted for the instances against their actual target values."
         },
        "meta":{
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[]
            }
  
        }
    

