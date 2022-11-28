import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import joblib
import h5py
import json
import shap
from flask_restful import Resource,reqparse
from flask import request
from saveinfo import save_file_info
from getmodelfiles import get_model_files


class ShapDeepGlobal(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder   

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id",required=True)
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        #getting params from info
        model_info=json.load(model_info_file)
        try:
            output_names=model_info["attributes"]["target_values"][0]
        except:
            output_names=None
        target_name=model_info["attributes"]["target_names"][0]
        feature_names=list(model_info["attributes"]["features"].keys())
        feature_names.remove(target_name)

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) 
            dataframe.drop([target_name], axis=1, inplace=True)
        else:
            raise Exception("The training data file was not provided.")

        #loading model (.h5 file)
        model=h5py.File(model_file, 'w')
        model = tf.keras.models.load_model(model)

        #getting params from request
        index=1
        if "output_index" in params_json:
            index=int(params_json["output_index"]);

        #creating explanation
        explainer = shap.DeepExplainer(model,dataframe.to_numpy())
        shap_values = explainer.shap_values(dataframe.to_numpy())
        
        if(len(np.array(shap_values).shape)==3): #multiclass shape: (#_of_classes, #_of_instances,#_of_features)
            shap_values=shap_values[index]
           
        #plotting
        plt.switch_backend('agg')
        shap.summary_plot(shap_values,features=dataframe,feature_names=feature_names,class_names=output_names)

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        plt.savefig(upload_folder+filename+".png",bbox_inches="tight")
       
        #formatting json output
        shap_values = [x.tolist() for x in shap_values]
        ret=json.loads(json.dumps(shap_values))
        
        response={"plot_png":getcall+".png","explanation":ret}

        return response


    def get(self):
        return {
        "_method_description": "This method based on Shapley values computes the average contribution of each feature for the whole training dataset. DeepSHAP is intended for TensorFlow/Keras models only. This method accepts 2 arguments: " 
                           "the 'id', and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "output_index": "(Optional) Integer representing the index of the class to be explained. Ignore for regression models. Defaults to class 1.",
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
    

