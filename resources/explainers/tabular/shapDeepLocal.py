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


class ShapDeepLocal(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder  

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id",required=True)
        parser.add_argument('params',required=True)
        args = parser.parse_args()
        
        _id = args.get("id")
        params_json = json.loads(args.get("params"))
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        #getting params from request
        instance = params_json["instance"]
        index=1
        plot_type=None
        if "output_index" in params_json:
            index=params_json["output_index"];
        if "plot_type" in params_json:
            plot_type=params_json["plot_type"];

        ##getting params from info
        model_info=json.load(model_info_file)

        kwargsData = dict(feature_names=None, output_names=None)

        if "feature_names" in model_info:
            kwargsData["feature_names"] = model_info["feature_names"]
        else:
            kwargsData["feature_names"]=["Feature "+str(i) for i in range(len(instance))]
        '''
        if "output_names" in model_info:
           kwargsData["output_names"] = model_info["output_names"]
        '''

        #load model (.h5 file)
        model=h5py.File(model_file, 'w')
        model = tf.keras.models.load_model(model)

        ## loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
            dataframe=dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False)
        else:
            raise Exception("The training data file was not provided.")

        # Create explanation
        explainer = shap.DeepExplainer(model,dataframe.to_numpy())
        shap_values = explainer.shap_values(np.array([instance]))

        if(len(np.array(shap_values).shape)!=1 and np.array(shap_values).shape[0]!=1):
            explainer.expected_value=explainer.expected_value[index]
            shap_values=shap_values[index]
           
        #plotting
        plt.switch_backend('agg')
        if plot_type=="bar":
            shap.plots._bar.bar_legacy(shap_values[0],features=np.array(instance),feature_names=kwargsData["feature_names"],show=False)
        elif plot_type=="decision":
            shap.decision_plot(explainer.expected_value,shap_values=shap_values[0],features=np.array(instance),feature_names=kwargsData["feature_names"])
        else:
            if plot_type==None:
                print("No plot type was specified. Defaulting to waterfall plot.")
            elif plot_type!="waterfall":
                print("No plot with the specified name was found. Defaulting to waterfall plot.")
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value,shap_values=shap_values[0],features=np.array(instance),feature_names=kwargsData["feature_names"],show=False)
       
        ##saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        plt.savefig(upload_folder+filename+".png",bbox_inches="tight")
       
        #formatting json output
        shap_values = [x.tolist() for x in shap_values]
        ret=json.loads(json.dumps(shap_values))
        
        #Insert code for image uploading and getting url
        response={"plot_png":getcall+".png","explanation":ret}

        return response


    def get(self):
        return {
        "_method_description": "This explaining method displays the contribution of each attribute for an individual prediction based on Shapley values (for tree ensemble methods only). Supported for XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models. This method accepts 2 arguments: " 
                           "the 'id', and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",

        "params": { 
                "instance": "Array with the feature values of an instance without including the target class.",
                "output_index": "(Optional) Integer representing the index of the class to be explained. Ignore for regression models. The default index is 1.",
                "plot_type": "(Optional) String with the name of the plot to be generated. The supported plots are 'waterfall','decision' and 'bar'. Defaults to 'waterfall'."
                },

        "params_example":{
                "instance": [1966, 62, 8, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                "output_index": 2
               }
  
        }
    

