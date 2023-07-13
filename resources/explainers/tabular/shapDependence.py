from http.client import BAD_REQUEST
from flask_restful import Resource
import joblib
import json
from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.dashboard_components.shap_components import ShapDependenceComponent
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants
import traceback



class ShapDependence(Resource):

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
        params_json={}
        if "params" in params:
            params_json=params["params"]

        return self.explain(_id, params_json)


    def explain(self,_id,params_json):
        try:
            #getting model info, data, and file from local repository
            model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

            #loading data
            if data_file!=None:
                dataframe = joblib.load(data_file) ##error handling?
            else:
                return "The training data file was not provided.",BAD_REQUEST

            #getting params from info
            model_info=json.load(model_info_file)
            backend = model_info["backend"]
            target_name=model_info["attributes"]["target_names"][0]
            model_task = model_info["model_task"]  
            features=list(dataframe.columns)
            features.remove(target_name)

            #loading model (.pkl file)
            if model_file!=None:
                if backend in ontologyConstants.SKLEARN_URIS:
                    model = joblib.load(model_file)
                elif backend in ontologyConstants.XGBOOST_URIS:
                    model = joblib.load(model_file)
                elif backend in ontologyConstants.LIGHTGBM_URIS:
                    model = joblib.load(model_file)
                else:
                    return "This explainer only supports scikit-learn-based models",BAD_REQUEST
            else:
                return "Model file was not uploaded.",BAD_REQUEST

            if model_task in ontologyConstants.CLASSIFICATION_URIS:
                explainer = ClassifierExplainer(model, dataframe.drop([target_name], axis=1, inplace=False), dataframe[target_name])
            elif model_task in ontologyConstants.REGRESSION_URIS:
                explainer = RegressionExplainer(model, dataframe.drop([target_name], axis=1, inplace=False), dataframe[target_name])
            else:
                return "AI task not supported. This explainer only supports scikit-learn-based classifiers or regressors.",BAD_REQUEST

            #getting params from request
            feature=None
            interaction_feature=None
            if "feature" in params_json:
                feat=str(params_json["feature"])
                if feat in features:
                    feature=feat
            if "interaction_feature" in params_json:
                feat=str(params_json["interaction_feature"])
                if feat in features:
                    interaction_feature=feat

            exp=ShapDependenceComponent(explainer,col=feature,color_col=interaction_feature)
            exp_html=exp.to_html().replace('\n', ' ').replace("\"","'")

            response={"type":"html","explanation":exp_html}
            return response
        except:
            return traceback.format_exc(), 500

    def get(self,id=None):
        
        base_dict={
        "_method_description": "Displays the the relationship between the values of a feature and the SHAP values. Only supports scikit-learn-based models. This method accepts 2 argument: " 
                           "the model 'id', and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "feature": {
                    "description":"Name of the feature to be displayed. Defaults to the feature with the highest average SHAP value.",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "interaction_feature":{
                    "description":"Name of the interaction feature used for coloring the instances. Defaults to the feature with the highest interaction value with the feature specified in the previous parameter.",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":False
                    } 
                },
        "output_description":{
                "dependence_plot": "The dependence plot the realationship between the values of a feature and the SHAP values. These values are colored according to the values of the interaction feature, displayed in the bar to the right."
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
                _, model_info_file, data_file = get_model_files(id,self.model_folder)
            except:
                return base_dict


            dataframe = joblib.load(data_file)
            model_info=json.load(model_info_file)
            target_name=model_info["attributes"]["target_names"][0]
            feature_names=list(dataframe.columns)
            feature_names.remove(target_name)

            base_dict["params"]["feature"]["default"]="Highest SHAP value"
            base_dict["params"]["feature"]["range"]=["Highest SHAP value"]+feature_names

            base_dict["params"]["interaction_feature"]["default"]="Highest Interact. value"
            base_dict["params"]["interaction_feature"]["range"]=["Highest Interact. value"]+feature_names

            return base_dict
           

        else:
            return base_dict
    

