from flask_restful import Resource,reqparse
import tensorflow as tf
#import torch
import numpy as np
import pandas as pd
import joblib
import werkzeug
import h5py
import json
import dice_ml

class DicePrivate(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        params_json = json.loads(args.get("params"))
        instance=params_json["instance"]
        features = params_json["features"]
        backend = params_json["backend"]
        num_cfs = params_json["num_cfs"]
        desired_class = params_json["desired_class"]
        method = params_json["method"]
        outcome_name = params_json["outcome_name"]
        features_to_vary = params_json["features_to_vary"]
        model = args.get("model")
      
        if backend=="TF1" or backend=="TF2":
            model =h5py.File(model, 'w')
            mlp = tf.keras.models.load_model(model)
        elif backend=="sklearn":
            mlp = joblib.load(model)
        else:
            mlp = torch.load(model)

        kwargsData = dict(features=features, outcome_name=outcome_name, type_and_precision=None, mad=None, data_name=None)

        if "type_and_precision" in params_json:
            kwargsData["type_and_precision"] = params_json["type_and_precision"]
        if "mad" in params_json:
            kwargsData["mad"] = params_json["mad"]
        if "data_name" in params_json:
            kwargsData["data_name"] = params_json["data_name"]

        # Create data
        d = dice_ml.Data(**{k: v for k, v in kwargsData.items() if v is not None})
  
        # Create model
        m = dice_ml.Model(model=mlp, backend=backend)

        # Create CFs generator using random
        exp = dice_ml.Dice(d, m, method=method)

       
        # Generate counterfactuals
        e1 = exp.generate_counterfactuals(instance, total_CFs=num_cfs, desired_class=desired_class, features_to_vary=features_to_vary)
        
        return json.loads(e1.cf_examples_list[0].final_cfs_df.to_json(orient='records'))

    def get(self):
        return {
        "_method_description": "Generates counterfactuals without the training data. However, it requires the format and ranges of the data. Currently supported for TensorFlow 2.0 models only. Requires 2 arguments: " 
                           "the 'params' string, and the 'model' which is a file containing the trained model.",

        "params": { "_description": "STRING representing a JSON object containing the following fields:",
                "instance": "JSON object representing the instance of interest with attribute names as keys, and feature values as values.",
                "backend": "A string containing the backend of the prediction model. Currently, the only supported backend for private data is 'TF2' (TensorFlow 2.0).",
                "method": "The method used for counterfactual generation. The supported methods for private data are: 'random' (random sampling) and 'genetic' (genetic algorithms).",
                "features": "JSON Object with feature names as keys and arrays containing the ranges of continuous features, or strings with the categories for categorical features.",
                "features_to_vary": "Either a string 'all' or a list of strings representing the feature names to vary.",
                "desired_class": "Integer representing the index of the desired counterfactual class, or 'opposite' in the case of binary classification.",
                "num_cfs": "number of counterfactuals to be generated for each instance.",
                "outcome_name": "name of the target column.",
                "type_and _precision": "(optional) JSON object with continuous feature names as keys. If the feature is of type int, the value should be the string 'int'. If the feature is of type float, an array of two values is expected, containing the string 'float', and the precision.",
                "mad": "(optional) JSON with feature names as keys and corresponding Median Absolute Deviation.",
                "data_name": "(optional) name of the dataset."
                },

        "params_example":{
                "backend": "TF2",
                "data_name": "datasetName",
                "features": {"Gender":["male", "female"], "Height": [ 0, 250], "Weight":[ 0, 250]},
                "features_to_vary": "all",
                "outcome_name": "Target",
                "desired_class": 0,
                "instances": [ ["X1", "X2", "Xn"], ["Y1", "Y2", "Yn"]],
                "method": "random",
                "num_cfs": 3,
                "type_and_precision": {"Height": ["float",1], "Weight": "int"}

               },

        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models. Currently, ONLY TensorFlow models are allowed for DicePrivate"
        }
