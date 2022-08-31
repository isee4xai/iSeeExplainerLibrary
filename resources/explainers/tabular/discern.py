from flask_restful import Resource, reqparse
import json
import pandas as pd
import numpy as np
from getmodelfiles import get_model_files
from saveinfo import save_file_info
import joblib
from html2image import Html2Image
from flask import request
from discern import discern_tabular

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

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        outcome_name="Target"
        categorical_features=[]
        if "target_name" in model_info:
            outcome_name = model_info["target_name"]
        try:
            feature_names = model_info["feature_names"]
        except:
            raise "The dataset \"features\" field was not specified."
        if "categorical_features" in model_info:
            categorical_features = model_info["categorical_features"]
      
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

        #getting params from request
        desired_class='opposite'
        feature_attribution_method='LIME'
        attributed_instance='Q'
        if "desired_class" in params_json:
            desired_class = params_json["desired_class"] if params_json["desired_class"]=="opposite" else int(params_json["desired_class"])
        if "feature_attribution_method" in params_json: 
            feature_attribution_method = params_json["feature_attribution_method"]  
        if "attributed_instance" in params_json:
            attributed_instance = params_json["attributed_instance"]

        ## init discern
        discern_obj = discern_tabular.DisCERNTabular(mlp, feature_attribution_method, attributed_instance)    
        dataframe_features = dataframe.loc[:, dataframe.columns != outcome_name].values
        dataframe_labels = dataframe[outcome_name].values
        target_values = dataframe[outcome_name].unique().tolist()
        discern_obj.init_data(dataframe_features, dataframe_labels, feature_names, target_values,**{'cat_feature_indices':categorical_features})

        cf, s, p = discern_obj.find_cf(instance, mlp.predict([instance])[0],desired_class)

        result_df = pd.DataFrame(np.array([instance, cf]), columns=feature_names)

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
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
        "_method_description": "Generates a counterfactual. Requires 3 arguments: " 
                           "the 'id' string, the 'instance' to be explained, and the 'params' object containing the configuration parameters of the explainer."
                           " These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array of feature values of the instance to be explained.",
        "params": { 
                
                "desired_class": "Integer representing the index of the desired counterfactual class, or 'opposite' in the case of binary classification. Defaults to 'opposite'.",
                "feature_attribution_method": "Feature attribution method used for obtaining feature weights; currently supports LIME, SHAP and Integrated Gradients. Defaults to LIME.",
                "attributed_instance": "Indicate on which instance to use feature attribution: Q for query or N for NUN. Defaults to Q."},

        "params_example":{
                "instance": [180, 78, 0.2],
                "desired_class": 'opposite',
                "feature_attribution_method": 'LIME',
                "attributed_instance": 'Q'},    
        }


