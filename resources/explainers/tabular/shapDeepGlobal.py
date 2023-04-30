import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import joblib
import h5py
import json
import shap
from flask_restful import Resource
from flask import request
from getmodelfiles import get_model_files
from io import BytesIO
from PIL import Image
from utils import ontologyConstants
from utils.base64 import PIL_to_base64


class ShapDeepGlobal(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder   

    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing"
        
        #Check params
        if("id" not in params):
            return "The model id was not specified in the params."

        _id =params["id"]
        if("type"  in params):
            inst_type=params["type"]
        url=None
        if "url" in params:
            url=params["url"]
        params_json={}
        if "params" in params:
            params_json=params["params"]
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        #getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        target_name=model_info["attributes"]["target_names"][0]
        output_names=model_info["attributes"]["features"][target_name]["values_raw"]
        feature_names=list(model_info["attributes"]["features"].keys())
        feature_names.remove(target_name)
        
        #getting params from request
        index=0
        if "target_class" in params_json:
            target_class=str(params_json["target_class"])
            try:
                index=output_names.index(target_class)
            except:
                pass

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        dataframe.drop([target_name], axis=1, inplace=True)

        #loading model (.h5 file)
        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                model=tf.keras.models.load_model(model)
            else:
                return "The model backend is not supported: " + backend
        else:
            return "Model file was not uploaded."

        #creating explanation
        explainer = shap.DeepExplainer(model,dataframe.to_numpy())
        shap_values = explainer.shap_values(dataframe.to_numpy())
        
        if(len(np.array(shap_values).shape)==3): #multiclass shape: (#_of_classes, #_of_instances,#_of_features)
            shap_values=shap_values[index]
           
        #plotting   
        plt.switch_backend('agg')
        shap.summary_plot(shap_values,features=dataframe, feature_names=feature_names,class_names=output_names,show=False)

        #formatting json output
        #shap_values = [x.tolist() for x in shap_values]
        #ret=json.loads(json.dumps(shap_values))

        ##saving
        img_buf = BytesIO()
        plt.savefig(img_buf,bbox_inches="tight")
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)
        plt.close()
        
        #Insert code for image uploading and getting url
        response={"type":"image","explanation":b64Image}

        return response


    def get(self):
        return {
        "_method_description": "This method based on Shapley values computes the average contribution of each feature for the whole training dataset. DeepSHAP is intended for TensorFlow/Keras models only. This method accepts 2 arguments: " 
                           "the 'id', and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "target_class": {
                    "description":"Name of the target class to be explained. Ignore for regression models. Defaults to the first class target class defined in the configuration file.",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":False
                    }
                },

        "output_description":{
                "beeswarm_plot": "The beeswarm plot is designed to display an information-dense summary of how the top features in a dataset impact the model's output. Each instance the given explanation is represented by a single dot" 
                                 "on each feature fow. The x position of the dot is determined by the SHAP value of that feature, and dots 'pile up' along each feature row to show density. Color is used to display the original value of a feature. "
               },
        "meta":{
                "supportsAPI":True,
                "needsData": True,
                "requiresAttributes":[]
            }
  
        }
    

