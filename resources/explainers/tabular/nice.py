from flask_restful import Resource
import tensorflow as tf
import torch
import pandas as pd
import joblib
import h5py
import json
import numpy as np
import requests
from nice import NICE
from PIL import Image
from io import BytesIO
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import PIL_to_base64
from utils.dataframe_processing import normalize_dict
import requests

from flask import request

class Nice(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
   
    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing."
        
        #Check params
        if("id" not in params):
            return "The model id was not specified in the params."
        if("type" not in params):
            return "The instance type was not specified in the params."
        if("instance" not in params):
            return "The instance was not specified in the params."

        _id =params["id"]
        if("type"  in params):
            inst_type=params["type"]
        instance=params["instance"]
        url=None
        if "url" in params:
            url=params["url"]
        params_json={}
        if "params" in params:
            params_json=params["params"]
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)



        #getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        target_name=model_info["attributes"]["target_names"][0]
        features=model_info["attributes"]["features"]
        output_names=features[target_name]["values_raw"]
        feature_names=list(features.keys())
        feature_names.remove(target_name)

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) 
        else:
            raise Exception("The training data file was not provided.")

        X=dataframe.drop([target_name], axis=1, inplace=False).values
        y=dataframe.loc[:,target_name].values

        categorical_features=[]
        for feature in feature_names:
            if features[feature]["data_type"]=="categorical":
                categorical_features.append(dataframe.columns.get_loc(feature))


        ## getting predict function
        predic_func=None
        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend in ontologyConstants.SKLEARN_URIS:
                mlp = joblib.load(model_file)
                try:
                    predic_func=mlp.predict_proba
                except:
                    predic_func=mlp.predict
            elif backend in ontologyConstants.PYTORCH_URIS:
                mlp = torch.load(model_file)
                predic_func=mlp.predict
            else:
                try:
                    mlp = joblib.load(model_file)
                    predic_func=mlp.predict
                except Exception as e:
                    return "Could not extract prediction function from model: " + str(e)
        elif url!=None:
            def predict(X):
                return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
            predic_func=predict
        else:
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")

        #normalize instance
        norm_instance=np.array(list(normalize_dict(instance,model_info).values()))

        instance_pred=np.array(predic_func(np.expand_dims(norm_instance,axis=0)))[0]
        print(instance_pred)

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
        NICE_res = NICE(predic_func,X,categorical_features,y_train=y,optimization=optimization_criteria)
        CF = NICE_res.explain(np.expand_dims(norm_instance,axis=0),target_class=desired_class)[0]

        print(norm_instance.shape)
        instance_row=np.array(np.append(norm_instance,np.argmax(instance_pred)))
        cf_row=np.array(list(CF)+[np.argmax(predic_func([CF])[0])])

        print(instance_row.shape)
        print(cf_row.shape)

        df = pd.DataFrame(data = np.array([instance_row,cf_row]), 
                  index = ["Original Instance","Counterfactual"], 
                  columns = feature_names + [target_name])

        df[target_name]=df[target_name].map(lambda x: output_names[int(x)])

        #saving
        #upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        #file = open(upload_folder+filename+".html", "w")
        #file.write(df.to_html())
        #file.close()
        #hti = Html2Image()
        #hti.output_path= upload_folder
        #if "png_height" in params_json and "png_width" in params_json:
        #    size=(int(params_json["png_width"]),int(params_json["png_height"]))
        #    hti.screenshot(html_str=df.to_html(), save_as=filename+".png",size=size)
        #else:
        #    hti.screenshot(html_str=df.to_html(), save_as=filename+".png")
        

        response={"type":"html","explanation":df.to_html()}
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
                },
        "output_description":{
                "html_table": "An html page containing a table with the original instance compared against the generated counterfactual."
               },
        "meta":{
                "supportsAPI":True,
                "needsData": True,
                "requiresAttributes":[]
            }
        }