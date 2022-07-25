from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import pandas as pd
import joblib
import h5py
import json
import dice_ml
from html2image import Html2Image
from saveinfo import save_file_info
from getmodelfiles import get_model_files
from flask import request

class DicePublic(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
   
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument("params")

        #parsing args
        args = parser.parse_args()
        _id = args.get("id")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        cont_features = model_info["cont_features"] ##error handling?

        ## loading model
        if backend=="TF1" or backend=="TF2":
           model_h5=h5py.File(model_file, 'w')
           model = tf.keras.models.load_model(model_h5)
        elif backend=="PYT":
           model = torch.load(model_file)
        else:
           model = joblib.load(model_file)
        
        ## loading data
        dataframe = joblib.load(data_file)
       

        ## params from the request
        kwargsData = dict(continuous_features=cont_features, outcome_name=dataframe.columns[-1])
        if "permitted_range" in params_json:
           kwargsData["permitted_range"] = params_json["permitted_range"]
        if "continuous_features_precision" in params_json:
           kwargsData["continuous_features_precision"] = params_json["continuous_features_precision"]

        kwargsData2 = dict(desired_class=1,total_CFs=3)
        if "num_cfs" in params_json:
           kwargsData2["total_CFs"] = params_json["num_cfs"]
        if "desired_class" in params_json:
           kwargsData2["desired_class"] = params_json["desired_class"]
        if "features_to_vary" in params_json:
           kwargsData2["features_to_vary"] = params_json["features_to_vary"]

        # Create data
        d = dice_ml.Data(dataframe=dataframe, **{k: v for k, v in kwargsData.items() if v is not None})
  
        # Create model
        m = dice_ml.Model(model=model, backend=backend)

        # Create CFs generator
        method="random"
        if "method" in params_json:
           method = params_json["method"]
        exp = dice_ml.Dice(d, m, method=method)

        columns = list(dataframe.columns)
        columns.remove(dataframe.columns[-1])

        instance = pd.DataFrame([instance], columns=columns)
       
        # Generate counterfactuals
        if backend=="sklearn":
            e1 = exp.generate_counterfactuals(query_instances=instance, **{k: v for k, v in kwargsData2.items() if v is not None})
        else:
            e1 = exp.generate_counterfactuals(instance, **{k: v for k, v in kwargsData2.items() if v is not None})
        
        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
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
        "_method_description": "Generates counterfactuals using the training data as a baseline. Accepts 3 arguments: " 
                           "the 'id' string, the 'instance', and the 'params' dictionary (optional) containing the configuration parameters of the explainer."
                           " These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance including the target class.",
        "params": { 
                "desired_class": "(optional) Integer representing the index of the desired counterfactual class. Defaults to class 1.  You may also use the string 'opposite' for binary classification",
                "method": "(optional) The method used for counterfactual generation. The supported methods are: 'random' (random sampling), 'genetic' (genetic algorithms), 'kdtrees'.  Defaults to 'random'.",
                "features_to_vary": "(optional) Either a string 'all' or a list of strings representing the feature names to vary. Defaults to all features.",
                "num_cfs": "(optional) number of counterfactuals to be generated for each instance.",
                "permitted_range": "(optional) JSON object with feature names as keys and permitted range in array as values.",
                "continuous_features_precision": "(optional) JSON object with feature names as keys and precisions as values."
                },
        "params_example":{
                "desired_class": 0,
                "features_to_vary": "all",
                "method": "random",
                "num_cfs": 3,
                "permitted_range": {"Height": [ 0, 250]},
                "continuous_features_precision": {"Height": 1, "Weight":3},
               }
        }