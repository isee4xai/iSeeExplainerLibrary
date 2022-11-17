from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import pandas as pd
import joblib
import h5py
import json
import numpy as np
import requests
from nice import NICE
from html2image import Html2Image
from saveinfo import save_file_info
from getmodelfiles import get_model_files
from flask import request

class Nice(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
   
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument("url")
        parser.add_argument("params")

        #parsing args
        args = parser.parse_args()
        _id = args.get("id")
        instance = json.loads(args.get("instance"))
        url = args.get("url")
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## loading data
        dataframe = joblib.load(data_file)
        X=dataframe.drop(dataframe.columns[-1],axis=1).values
        y=dataframe.iloc[:,-1].values

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?

        output_names=None
        if "feature_names" in model_info:
            feature_names=model_info["feature_names"]
        else:
            try:
                feature_names=list(dataframe.drop(dataframe.columns[-1],axis=1).columns)
            except:
                feature_names=["Feature "+str(i) for i in range(len(instance))]
        if "categorical_features" in model_info:    #necessary                        
            categorical_features=model_info["categorical_features"]
        else:
            raise Exception("Array of categorical features must be specified.")
        if "output_names" in model_info:
            output_names = model_info["output_names"]
        if "target_name" in model_info:
            target_name = model_info["target_name"]
        else:
            try:
                target_name=dataframe.columns[-1]
            except:
                target_name="Target"

        ## getting predict function
        predic_func=None
        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend=="sklearn":
                mlp = joblib.load(model_file)
                try:
                    predic_func=mlp.predict_proba
                except:
                    predic_func=mlp.predict
            elif backend=="PYT":
                mlp = torch.load(model_file)
                predic_func=mlp.predict
            else:
                mlp = joblib.load(model_file)
                predic_func=mlp.predict
        elif url!=None:
            def predict(X):
                return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
            predic_func=predict
        else:
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")

        instance_pred=np.array(predic_func([instance]))[0]

        ## params from the request
        optimization_criteria="sparsity"
        desired_class="other"
        if "optimization_criteria" in params_json and params_json["optimization_criteria"] in ["sparsity","proximity","plausibility"]:
           optimization_criteria = params_json["optimization_criteria"]
        if "desired_class" in params_json:
           try: 
               u_class=int(params_json["desired_class"])
               if u_class >= 0 and u_class < instance_pred.shape[-1]:
                   desired_class=[u_class]
           except:
               pass

        # Generate counterfactuals
        NICE_adult = NICE(predic_func,X,categorical_features,y_train=y,optimization=optimization_criteria)
        CF = NICE_adult.explain(np.array([instance]),target_class=desired_class)[0]

        instance_row=np.array(instance+[np.argmax(instance_pred)])
        cf_row=np.array(list(CF)+[np.argmax(predic_func([CF])[0])])

        df = pd.DataFrame(data = np.array([instance_row,cf_row]), 
                  index = ["Original Instance","Counterfactual"], 
                  columns = feature_names + [target_name])

        if output_names is not None:
            df[target_name]=df[target_name].map(lambda x: output_names[int(x)])

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        file = open(upload_folder+filename+".html", "w")
        file.write(df.to_html())
        file.close()
        hti = Html2Image()
        hti.output_path= upload_folder
        hti.screenshot(html_str=df.to_html(), save_as=filename+".png")

        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":df.to_dict()}
        return response

    def get(self):
        return {
        "_method_description": "NICE is an algorithm to generate Counterfactual Explanations for heterogeneous tabular data."
                               "NICE exploits information from a nearest instance to speed up the search process and guarantee that an explanation will be found. Accepts 4 arguments: " 
                           "the 'id' string, the 'instance', the 'url' (optional), and the 'params' dictionary (optional) containing the configuration parameters of the explainer."
                           " These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance not including the target class.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "desired_class": "(optional) Integer representing the index of the desired counterfactual class. Defaults to string 'other', which will look for any different class.",
                "optimization_criteria": "(optional) The counterfactual criteria to optimize. It can be 'sparsity','proximity', or 'plausibility'. Defaults to 'sparsity.'"
                }
        }