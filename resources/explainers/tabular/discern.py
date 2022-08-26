from flask_restful import Resource, reqparse
import json
import pandas as pd
import numpy as np
from getmodelfiles import get_model_files
from saveinfo import save_file_info
import joblib
from html2image import Html2Image
import h5py
import tensorflow as tf
import torch
from flask import request

import discern.discern_tabular

class DisCERN(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument("params",required=True)

        #parsing args
        args = parser.parse_args()
        _id = args.get("id")
        params=args.get("params")
        params_json=json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        outcome_name="Target"
        if "target_name" in model_info:
            outcome_name = model_info["target_name"]
        try:
            features = model_info["features"]
        except:
            raise "The dataset \"features\" field was not specified."
      
        ## loading model
        # if backend=="TF1" or backend=="TF2":
        #     model=h5py.File(model_file, 'w')
        #     mlp = tf.keras.models.load_model(model)
        #     predic_func=mlp
        # elif backend=="sklearn":
        #     mlp = joblib.load(model_file)
        #     predic_func=mlp.predict_proba
        # elif backend=="PYT":
        #     mlp = torch.load(model_file)
        #     predic_func=mlp.predict
        # else:
        #     mlp = joblib.load(model_file)
        #     predic_func=mlp.predict
        try:
            # if backend=="sklearn":
            mlp = joblib.load(model_file)
            # predic_func=mlp.predict_proba
        except:
            raise "Currently only supports sklearn models"

        ## loading data
        dataframe = joblib.load(data_file)

        try:
            instance = params_json["instance"]
        except:
            raise "No instance was provided in the params."

        ## init discern
        discern = discern.discern_tabular.DisCERNTabular(mlp, params_json["feature_attribution_method"], params_json["attributed_instance"])    
        dataframe_features = dataframe.loc[:, dataframe.columns != outcome_name].values
        dataframe_labels = dataframe[outcome_name].values
        target_values = dataframe[outcome_name].unique()
        discern.init_data(dataframe_features, dataframe_labels, features, target_values)

        cf, s, p = discern.find_cf(instance, mlp.predict([instance])[0])

        result_df = pd.DataFrame(np.array([instance, cf]), columns=features)

        #saving
        upload_folder, filename, getcall = save_file_info(request.path)
        str_html= result_df.to_html()+'<br>'
        
        file = open(upload_folder+filename+".html", "w")
        file.write(str_html)
        file.close()

        hti = Html2Image()
        hti.output_path= upload_folder
        hti.screenshot(html_str=str_html, save_as=filename+".png")
        
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":result_df.to_json()}
        return response

    def get(self):
        return {
        "_method_description": "Generates a counterfactual. Requires 2 arguments: " 
                           "the 'id' string, and the 'params' object containing the configuration parameters of the explainer."
                           " These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",

        "params": { 
                "instance": "Array of feature values of the instance",
                "desired_class": "Integer representing the index of the desired counterfactual class, or 'opposite' in the case of binary classification.",
                "feature_attribution_method": "Feature attribution method used for obtaining feature weights; currently supports LIME, SHAP and Integrated Gradients",
                "attributed_instance": "Indicate on which instance to use feature attribution: Q for query or N for NUN"},

        "params_example":{
                "instance": {180, 78, 0.2},
                "desired_class": 'opposite',
                "feature_attribution_method": "LIME",
                "attributed_instance": 'Q'},    
        }


