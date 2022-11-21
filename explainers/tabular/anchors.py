from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
from alibi.explainers import AnchorTabular
from getmodelfiles import get_model_files
import requests

class Anchors(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id",required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument("url")
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?

        kwargsData = dict(feature_names=None,categorical_names=None, ohe=False)
        if "feature_names" in model_info:
            kwargsData["feature_names"]=model_info["feature_names"]
        if "categorical_names" in model_info:
            cat_names = model_info["categorical_names"]
            kwargsData["categorical_names"] = {int(k):v for k,v in cat_names.items()}
        if "ohe" in model_info:
            kwargsData["ohe"] = bool(model_info["ohe"])

        ## getting predict function
        predic_func=None
        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend=="sklearn":
                mlp = joblib.load(model_file)
                predic_func=mlp.predict_proba
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

        #getting params from request
        kwargsData2 = dict(threshold=0.95)
        if "threshold" in params_json:
            kwargsData2["threshold"] = float(params_json["threshold"])

        # Create data
        explainer = AnchorTabular(predic_func, **{k: v for k, v in kwargsData.items()})

        explainer.fit(dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False).to_numpy(), disc_perc=(25, 50, 75))
        
        explanation = explainer.explain(np.array(instance), **{k: v for k, v in kwargsData2.items()})
        
        if explanation.anchor:
            ret = dict(anchor=(' AND '.join(explanation.anchor)),precision=explanation.precision, coverage=explanation.coverage)
        else:
            ret = dict(anchor=(' AND '.join(explanation.anchor)),precision=explanation.precision[0], coverage=explanation.coverage)
        return json.loads(json.dumps(ret))


    def get(self):
        return {
        "_method_description": "Anchors provide local explanations in the form of simple boolean rules with a precision score and a "
                            "coverage value which represents the scope in which that rules applies to similar instances. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url' (optional),  and the 'params' JSON (optional) with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance without including the target class.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",

        "params": { 
                "threshold": "(Optional) The minimum level of precision required for the anchors. Default is 0.95"
                },
        "output_description":{
                "anchor_json": "A JSON object with the boolean rule (anchor) that was found, and values for its precision and coverage (scope in which that rules applies to similar instances)."
               }
        }
    
