from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import joblib
import werkzeug
import h5py
import json
import dice_ml
from html2image import Html2Image
from saveinfo import save_file_info
from flask import request

class DicePublic(Resource):
   
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("data", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        
        data = args.get("data")
        dataframe = joblib.load(data)
        params_json = json.loads(args.get("params"))
        instances=params_json["instances"]
        cont_features = params_json["cont_features"]
        backend = params_json["backend"]
        num_cfs = params_json["num_cfs"]
        desired_class = params_json["desired_class"]
        method = params_json["method"]
        features_to_vary = params_json["features_to_vary"]

        model = args.get("model")
        
        if backend=="TF1" or backend=="TF2":
            model=h5py.File(model, 'w')
            mlp = tf.keras.models.load_model(model)
        elif backend=="sklearn":
            mlp = joblib.load(model)
        else:
            mlp = torch.load(model)
      
        kwargsData = dict(continuous_features=cont_features, outcome_name=dataframe.columns[-1], permitted_range=None, continuous_features_precision=None, data_name=None)

        if "permitted_range" in params_json:
            kwargsData["permitted_range"] = params_json["permitted_range"]
        if "continuous_features_precision" in params_json:
            kwargsData["continuous_features_precision"] = params_json["continuous_features_precision"]


        # Create data
        d = dice_ml.Data(dataframe=dataframe, **{k: v for k, v in kwargsData.items() if v is not None})
  
        # Create model
        m = dice_ml.Model(model=mlp, backend=backend)

        # Create CFs generator using random
        exp = dice_ml.Dice(d, m, method=method)

        columns = list(dataframe.columns)
        columns.remove(dataframe.columns[-1])

        instances = pd.DataFrame(instances, columns=columns)
       
        # Generate counterfactuals
        if backend=="sklearn":
            e1 = exp.generate_counterfactuals(query_instances=instances, total_CFs=num_cfs, desired_class=desired_class, features_to_vary=features_to_vary)
        else:
            e1 = exp.generate_counterfactuals(instances, total_CFs=num_cfs, desired_class=desired_class, features_to_vary=features_to_vary)
        
        #saving
        upload_folder, filename, getcall = save_file_info(request.path)
        str_html=''
        i=1
        for cf in e1.cf_examples_list:
            if cf.final_cfs_df is None:
                cfs = "<h3>No counterfactuals were found for this instance. Perhaps try with different features.</h3>"    
           
            else:
                cfs=cf.final_cfs_df.to_html()

            str_html =  str_html + '<h2>Instance ' + str(i) + '</h2>' + cf.test_instance_df.to_html() + '<h2>Counterfactuals</h2>'+ cfs + '<br><br><hr><br>'
            i=i+1

        file = open(upload_folder+filename+".html", "w")
        file.write(str_html)
        file.close()

        hti = Html2Image()
        hti.output_path= upload_folder
        hti.screenshot(html_str=str_html, save_as=filename+".png")

        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":json.loads(e1.to_json())}
        return response

    def get(self):
        return {
        "_method_description": "Generates counterfactuals using the training data as a baseline. Requires 3 arguments: " 
                           "the 'params' string, the 'model' which is a file containing the trained model, and " 
                           "the 'data', containing the training data used for the model. These arguments are described below.",

        "params": { "_description": "STRING representing a JSON object containing the following fields:",
                "instances": "Array of arrays, where each one represents a row with the feature values of an instance including the target class.",
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "method": "The method used for counterfactual generation. The supported methods are: 'random' (random sampling), 'genetic' (genetic algorithms), 'kdtrees'.",
                "cont_features": "Array of strings containing the name of the continuous features. Features not included here are considered categorical.",
                "features_to_vary": "Either a string 'all' or a list of strings representing the feature names to vary.",
                "desired_class": "Integer representing the index of the desired counterfactual class, or 'opposite' in the case of binary classification.",
                "num_cfs": "number of counterfactuals to be generated for each instance.",
                "permitted_range": "(optional) JSON object with feature names as keys and permitted range in array as values.",
                "continuous_features_precision": "(optional) JSON object with feature names as keys and precisions as values."
                },

        "params_example":{
                "backend": "sklearn",
                "cont_features": ["Height", "Weight"],
                "continuous_features_precision": {"Height": 1, "Weight":3},
                "desired_class": 0,
                "features_to_vary": "all",
                "instances": [ ["X1", "X2", "Xn"], ["Y1", "Y2", "Yn"]],
                "method": "random",
                "num_cfs": 3,
                "permitted_range": {"Height": [ 0, 250]}
               },

        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models.",

        "data": "Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The target class must be the last column of the DataFrame"
        }