from flask_restful import Resource
import joblib
import json
import math
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from io import BytesIO
from PIL import Image
from flask import request
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import PIL_to_base64


class SklearnICE(Resource):

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
        target_names=model_info["attributes"]["target_names"]
        target_name=target_names[0]
        features=model_info["attributes"]["features"]

        output_names=[]
        if model_info["attributes"]["features"][target_name]["data_type"]=="categorical": #is classification
            output_names=model_info["attributes"]["features"][target_name]["values_raw"]

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        dataframe.drop(target_names,axis=1,inplace=True)

        categorical_features=[]
        for feature in dataframe.columns:
            if features[feature]["data_type"]=="categorical":
                categorical_features.append(True)
            else:
                categorical_features.append(False)

        print(categorical_features)

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

        #getting params from request
        features=None
        if "features_to_show" in params_json and params_json["features_to_show"]:
            features=json.loads(params_json["features_to_show"]) if isinstance(params_json["features_to_show"],str) else params_json["features_to_show"]
            features=[dataframe.columns.get_loc(c) for c in features if c in dataframe]

        target=None
        if "target_class" in params_json: # only present for multiclass settings
            target_class=str(params_json["target_class"])
            try:
                target=output_names.index(target_class)
            except:
                pass

        if(len(output_names)>2 and target is None): #multiclass
            target=1

        if(features is None):
            features=[i for i in range(len(dataframe.columns))] #defaults to all features

        fig, ax = plt.subplots(figsize=(18,math.ceil(len(features)/3)*6))
        features=[f for f in features if not categorical_features[f]]

        if(not features):
            return {"type":"text","explanation":"ICE can only be plotted for continuous features and none were found."}

        print([list(dataframe.columns)[f] for f in features])
        PartialDependenceDisplay.from_estimator(model,dataframe,features,categorical_features=categorical_features,feature_names=dataframe.columns,kind="both",ax=ax,target=target)

        #saving
        img_buf = BytesIO()
        plt.savefig(img_buf,bbox_inches="tight")
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response={"type":"html","explanation":b64Image}
        return response


    def get(self,id=None):
        
        base_dict={
        "_method_description": "Displays the Individual Conditional Expectation (ICE) plot for the specified features. Only supports scikit-learn-based models. This method accepts 2 arguments: " 
                           "the model 'id' and the 'params' object.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                    "features_to_show": {
                        "description":"Array of strings representing the name of the features to be explained. Defaults to all features.",
                        "type":"array",
                        "default": None,
                        "range":None,
                        "required":False
                        },
        "output_description":{
                "partial_dependence_plot": "Shows the dependence between the target function and an input feature of interest. However, unlike a PDP, which shows the average effect of the input feature, an ICE plot visualizes the dependence of the prediction on a feature for each sample separately with one line per sample."
         },

        "meta":{
                "modelAccess":"File",
                "supportsBWImage":False,
                "needsTrainingData": True
            }
  
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

            base_dict["params"]["features_to_show"]["default"]=feature_names
            base_dict["params"]["features_to_show"]["range"]=feature_names

            output_names=[]
            if model_info["attributes"]["features"][target_name]["data_type"]=="categorical": #is classification
                output_names=model_info["attributes"]["features"][target_name]["values_raw"]
                if(len(output_names)>2): #multiclass
                    target_class={
                        "description":"Name of the target class to be explained. Ignored for regression models. Defaults to the class at index 1 in the configuration file.",
                        "type":"string",
                        "default":output_names[1],
                        "range":output_names,
                        "required":False
                    }
                    base_dict["params"]["target_class"]=target_class


            return base_dict
           

        else:
            return base_dict
    

