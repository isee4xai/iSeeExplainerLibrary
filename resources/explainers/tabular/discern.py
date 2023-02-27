from flask_restful import Resource, reqparse
import json
import pandas as pd
import numpy as np
from getmodelfiles import get_model_files
from saveinfo import save_file_info
import joblib
import h5py
from html2image import Html2Image
from flask import request
from discern import discern_tabular
import tensorflow as tf

class DisCERN(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument("instance", required=True)
        parser.add_argument("params")

        #parsing args
        args = parser.parse_args()
        _id = args.get("id")
        params=args.get("params")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params!=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## loading data
        dataframe = joblib.load(data_file)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        outcome_name=model_info["attributes"]["target_names"][0]
        feature_names=list(model_info["attributes"]["features"].keys())
        feature_names.remove(outcome_name)
        categorical_features=[]
        for feature in feature_names:
            if isinstance(model_info["attributes"]["features"][feature],list):
                categorical_features.append(dataframe.columns.get_loc(feature))
      
        ## loading model
        if backend=="TF1" or backend=="TF2":
            model=h5py.File(model_file, 'w')
            model = tf.keras.models.load_model(model)
        elif backend=="sklearn":
            model = joblib.load(model_file)
        # elif backend=="PYT":
        #     mlp = torch.load(model_file)
        #     predic_func=mlp.predict
        else:
            raise Exception("Currently supports Tensoflow and scikit-learn classification models.")
        

        #getting params from request
        desired_class='opposite'
        feature_attribution_method='LIME'
        imm_features = []
        if "desired_class" in params_json:
            desired_class = params_json["desired_class"] if params_json["desired_class"]=="opposite" else int(params_json["desired_class"])
        if "feature_attribution_method" in params_json: 
            feature_attribution_method = params_json["feature_attribution_method"]  
        if "immutable_features" in params_json: 
            imm_features = params_json["immutable_features"]  

        ## init discern
        discern_obj = discern_tabular.DisCERNTabular(model, feature_attribution_method)    
        dataframe_features = dataframe.drop(dataframe.columns[-1],axis=1).values
        dataframe_labels = dataframe.iloc[:,-1].values
        target_values = dataframe[outcome_name].unique().tolist()
        discern_obj.init_data(dataframe_features, dataframe_labels, feature_names, target_values, **{'cat_feature_indices':categorical_features, 'immutable_feature_indices':imm_features})

        test_label = None 
        if backend=="TF1" or backend=="TF2":
            test_label = model.predict(np.array([instance])).argmax(axis=-1)[0]
        elif backend=="sklearn":
            test_label = model.predict([instance])[0]
        
        cf, cf_label, s, p = discern_obj.find_cf(instance, test_label, desired_class)

        result_df = pd.DataFrame(np.array([instance, cf]), columns=feature_names)

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        str_html= result_df.to_html()+'<br>'
        
        file = open(upload_folder+filename+".html", "w")
        file.write(str_html)
        file.close()

        hti = Html2Image()
        hti.output_path= upload_folder
        if "png_height" in params_json and "png_width" in params_json:
            size=(int(params_json["png_width"]),int(params_json["png_height"]))
            hti.screenshot(html_str=str_html, save_as=filename+".png",size=size)
        else:
            hti.screenshot(html_str=str_html, save_as=filename+".png")
       
        
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":result_df.to_json()}
        return response

    def get(self):
        return {
        "_method_description": "Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods (DisCERN) generates counterfactuals for scikit-learn-based models. Requires 3 arguments: " 
                           "the 'id' string, the 'instance' to be explained, and the 'params' object containing the configuration parameters of the explainer."
                           " These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array of feature values of the instance to be explained.",
        "params": {
                "desired_class": "Integer representing the index of the desired counterfactual class, or 'opposite' in the case of binary classification. Defaults to 'opposite'.",
                "feature_attribution_method": "Feature attribution method used for obtaining feature weights; currently supports LIME, SHAP and Integrated Gradients. Defaults to LIME.",
                "immutable_features": "Array of feature indices that are immutable. The counterfactual will not recommend to change these features. Default is an empty array",
                },

        "output_description":{
                "html_table": "An html page containing a table with the generated couterfactuals."
               },
        "meta":{
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[]
            }
        }


