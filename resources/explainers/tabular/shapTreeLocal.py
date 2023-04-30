from flask_restful import Resource
import numpy as np
import joblib
import json
import shap
from flask import request
import matplotlib.pyplot as plt
from getmodelfiles import get_model_files
from io import BytesIO
from PIL import Image
from utils import ontologyConstants
from utils.base64 import PIL_to_base64
from utils.dataframe_processing import normalize_dict


class ShapTreeLocal(Resource):

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
        params_json={}
        if "params" in params:
            params_json=params["params"]
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        #getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        target_name=model_info["attributes"]["target_names"][0]
        output_names=model_info["attributes"]["features"][target_name]["values_raw"]
        feature_names=list(model_info["attributes"]["features"].keys())
        feature_names.remove(target_name)
        kwargsData = dict(feature_names=feature_names, output_names=output_names)

        #getting params from request
        index=0
        if "target_class" in params_json:
            target_class=str(params_json["target_class"])
            try:
                index=output_names.index(target_class)
            except:
                pass
        plot_type=None
        if "plot_type" in params_json:
            plot_type=params_json["plot_type"]

        #loading model (.pkl file)
        if model_file!=None:
            if backend in ontologyConstants.SKLEARN_URIS:
                model = joblib.load(model_file)
            elif backend in ontologyConstants.XGBOOST_URIS:
                model = joblib.load(model_file)
            elif backend in ontologyConstants.LIGHTGBM_URIS:
                model = joblib.load(model_file)
            else:
                return "The model backend is not supported: " + backend
        else:
            return "Model file was not uploaded."

        
        #normalize instance
        norm_instance=np.array(list(normalize_dict(instance,model_info).values()))

        # Create explanation
        explainer = shap.Explainer(model,**{k: v for k, v in kwargsData.items()})
        shap_values = explainer.shap_values(np.expand_dims(norm_instance,axis=0))

        if(len(np.array(shap_values).shape)==3):
            explainer.expected_value=explainer.expected_value[index]
            shap_values=shap_values[index]
           
        #plotting
        plt.switch_backend('agg')
        if plot_type=="bar":
            shap.plots._bar.bar_legacy(shap_values[0],features=np.array(list(instance.values())),feature_names=kwargsData["feature_names"],show=False)
        elif plot_type=="decision":
            shap.decision_plot(explainer.expected_value,shap_values=shap_values[0],features=np.array(list(instance.values())),feature_names=kwargsData["feature_names"])
        elif plot_type=="force":
                shap.plots._force.force(explainer.expected_value,shap_values=shap_values[0],features=np.array(list(instance.values())),feature_names=kwargsData["feature_names"],out_names=target_name,matplotlib=True,show=False)
        else:
            if plot_type==None:
                print("No plot type was specified. Defaulting to waterfall plot.")
            elif plot_type!="waterfall":
                print("No plot with the specified name was found. Defaulting to waterfall plot.")
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value,shap_values=shap_values[0],features=np.array(list(instance.values())),feature_names=kwargsData["feature_names"],show=False)
       
        #formatting json output
        #shap_values = [x.tolist() for x in shap_values]
        #ret=json.loads(json.dumps(shap_values))

        ##saving
        img_buf = BytesIO()
        plt.savefig(img_buf,bbox_inches="tight")
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)
        
        response={"type":"image","explanation":b64Image}

        return response



    def get(self):
        return {
        "_method_description": "This method displays the contribution of each attribute for an individual prediction based on Shapley values (for tree ensemble methods only). Supported for XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models. This method accepts 3 arguments: " 
                           "the 'id', the 'instance', and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array with the feature values of an instance without including the target class.",
        "params": { 
                "target_class": {
                    "description":"Name of the target class to be explained. Ignore for regression models. Defaults to the first class target class defined in the configuration file.",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "plot_type": {
                    "description":"String with the name of the plot to be generated.",
                    "type":"string",
                    "default": "waterfall",
                    "range":['waterfall','decision','force','bar'],
                    "required":False
                    }
                },
        "output_description":{
                "waterfall_plot": "Waterfall plots are designed to display explanations for individual predictions, so they expect a single row of an Explanation object as input. "
                                    "The bottom of a waterfall plot starts as the expected value of the model output, and then each row shows how the positive (red) or negative (blue) contribution of "
                                    "each feature moves the value from the expected model output over the background dataset to the model output for this prediction.",
                "force_plot":"Displays the contribution of each attribute as a plot that confronts the features that contribute positively (left) and the ones that contribute negatively (right) to the predicted outcome. "
                             "The predicted outcome is displayed as a divisory line between the positive and negative contributions.",

                "decision_plot": "A decision plot shows how a complex model arrive at its predictions. "
                                "The decision plot displays the average of the model's base values and shifts the SHAP values accordingly to accurately reproduce the model's scores."
                                "The straight vertical line marks the model's base value. The colored line is the prediction. Feature values are printed next to the prediction line for reference."
                                "Starting at the bottom of the plot, the prediction line shows how the SHAP values (i.e., the feature effects) accumulate from the base value to arrive at the model's final score at the top of the plot. ",

                "bar_plot": "The bar plot is a local feature importance plot, where the bars are the SHAP values for each feature. Note that the feature values are shown in the left next to the feature names."
         },
        "meta":{
                "supportsAPI":True,
                "needsData": False,
                "requiresAttributes":[]
            }
  
        }
    

