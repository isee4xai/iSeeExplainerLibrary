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
        parser.add_argument('id',required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        #getting params from request
        index=1
        plot_type=None
        if "output_index" in params_json:
            index=int(params_json["output_index"]);
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
        "_method_description": "This method displays the contribution of each attribute for an individual prediction based on Shapley values (for tree ensemble methods only). Supported for XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models. This method accepts 3 arguments: " 
                           "the 'id', the 'instance', and the 'params' dictionary (optional) with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array with the feature values of an instance without including the target class.",
        "params": { 
                "output_index": "(Optional) Integer representing the index of the class to be explained. Ignore for regression models. The default index is 1.",
                "plot_type": "(Optional) String with the name of the plot to be generated. The supported plots are 'waterfall','decision' and 'bar'. Defaults to 'waterfall'."
                },
        "output_description":{
                "waterfall_plot": "Waterfall plots are designed to display explanations for individual predictions, so they expect a single row of an Explanation object as input. "
                                    "The bottom of a waterfall plot starts as the expected value of the model output, and then each row shows how the positive (red) or negative (blue) contribution of "
                                    "each feature moves the value from the expected model output over the background dataset to the model output for this prediction.",

                "decision_plot": "A decision plot shows how a complex model arrive at its predictions. "
                                "The decision plot displays the average of the model's base values and shifts the SHAP values accordingly to accurately reproduce the model's scores."
                                "The straight vertical line marks the model's base value. The colored line is the prediction. Feature values are printed next to the prediction line for reference."
                                "Starting at the bottom of the plot, the prediction line shows how the SHAP values (i.e., the feature effects) accumulate from the base value to arrive at the model's final score at the top of the plot. ",

                "bar_plot": "The bar plot is a local feature importance plot, where the bars are the SHAP values for each feature. Note that the feature values are shown in the left next to the feature names."
         }
        }
    

