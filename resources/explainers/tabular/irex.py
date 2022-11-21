from flask_restful import Resource,reqparse
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
from alibi.explainers import ALE
import matplotlib.pyplot as plt
import seaborn as sns
from flask import request
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests

class IREX(Resource):
    
    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id",required=True)
        parser.add_argument("url")
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
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
        model_task=model_info["model_task"] ##error handling?

        if model_task!="classification":
            raise Exception("IREX can only be used with classification models.")

        try:
            expected=model_info["expected_answers"]
        except:
            raise Exception("An array with the expected answers is necessary for the IREX explainer.")

        kwargsData = dict(feature_names=None,target_names=None)
        if "feature_names" in model_info:
            kwargsData["feature_names"]=model_info["feature_names"]
        if "output_names" in model_info:
            kwargsData["target_names"]=model_info["output_names"]

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
      
        #getting params from request
        threshold=0.01
        classes_to_show=None
        if "threshold" in params_json:
            threshold = float(params_json["threshold"])
        if "classes_to_show" in params_json:
            classes_to_show = json.loads(params_json["classes_to_show"]) if isinstance(params_json["classes_to_show"],str) else params_json["classes_to_show"]


        proba_ale_lr = ALE(predic_func, **{k: v for k, v in kwargsData.items()})
        proba_exp_lr = proba_ale_lr.explain(dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False).to_numpy())

        anomalies=[]
        for i in range(len(proba_exp_lr.ale_values)):
            if(proba_exp_lr.ale_values[i][expected[i]][-1]<0-threshold):
                anomalies.append(i)

        arranged_ale=[]
        if classes_to_show is None:
            classes_to_show=range(proba_exp_lr.ale_values[0].shape[-1])
        i=0
        for oclass in classes_to_show:
            arranged_ale.append([])
            for q in range(len(proba_exp_lr.ale_values)):
                arranged_ale[i].append(abs(proba_exp_lr.ale_values[q][expected[q]][oclass]))
            i=i+1

        ale_df=pd.DataFrame(arranged_ale)
        if kwargsData["target_names"] is None:
            class_names=["Class " +str(i) for i in classes_to_show]
        else:
            class_names=[kwargsData["target_names"][oclass] for oclass in classes_to_show]
        ale_df.index=class_names
        if kwargsData["feature_names"] is None:
            kwargsData["feature_names"]=range(1,len(proba_exp_lr.ale_values)+1)
        ale_df.columns=kwargsData["feature_names"]
        ale_df=ale_df.transpose()

        df_mask = [[False for _ in range(len(proba_exp_lr.ale_values))]]
        for ind in anomalies:
            df_mask[0][ind] = True
        df_mask = pd.DataFrame(df_mask)
        df_masks=[df_mask for _ in classes_to_show]
        mask=pd.concat(df_masks, ignore_index = True, axis = 0)
        plt.figure(figsize = (len(proba_exp_lr.ale_values)/2,len(classes_to_show)*3))
        sns.heatmap(ale_df.transpose(), cmap="Reds", cbar=False)
        sns.heatmap(ale_df.transpose(), cmap="Blues",yticklabels=True, xticklabels=True,mask=mask.to_numpy())

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        plt.savefig(upload_folder+filename+".png",bbox_inches='tight')

        proba_exp_lr["meta"]["name"]="IREX-ALE"
        response = {"plot_png":getcall+'.png',"explanation":json.loads(proba_exp_lr.to_json())}
        return response

    def get(self):
        return {
        "_method_description": "IREX is a reusable method for the Iterative Refinement and EXplanation of classification models. It has been designed for domain-expert users -without machine learning skills- that need to understand" 
        " and improve classification models. This particular implementation of IREX uses ALE to identify anomalous features that may be contradictory to what the expert knowledge indicates. Anomalous features are highlighted in red in an ALE heatmap. This method accepts 3 arguments: " 
                           "the 'id', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "threshold": "(Optional) A float between 0 and 1 for the threshold that will be used to determine anomalous variables. If a feature seems to be contradictory but its absolute ALE value is below this threshold, it will not be considered anomalous. Defaults to 0.01.",
                "classes_to_show": "(Optional) Array of ints representing the indices of the classes to be explained. Defaults to all classes."
                },
        "output_description":{
                "heatmap": "A heatmap displaying the relevance of the features according to ALE, where anomalous features (behavior differring from expected values) are highlighted in red."
               },
        "meta":{
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[{"expected_answers":"Array containing the expected answers to the questions of a questionnaire that are supposed to contribute to the target class by experts." }]
            }
        }
    
