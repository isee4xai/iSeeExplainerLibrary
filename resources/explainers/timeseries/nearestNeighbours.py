from http.client import BAD_REQUEST
from flask_restful import Resource
from flask import request
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import torch
import plotly.express as px
import h5py
import json
import joblib
import requests
from tslearn.neighbors import KNeighborsTimeSeries
from getmodelfiles import get_model_files
from utils import ontologyConstants
import traceback




class TSNearestNeighbours(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder 
        
    def post(self):
        try:
            params = request.json
            if params is None:
                return "The json body is missing.",BAD_REQUEST
        
            #Check params
            if("id" not in params):
                return "The model id was not specified in the params.",BAD_REQUEST
            if("type" not in params):
                return "The instance type was not specified in the params.",BAD_REQUEST
            if("instance" not in params):
                return "The instance was not specified in the params.",BAD_REQUEST

            _id =params["id"]
            if("type"  in params):
                inst_type=params["type"]
            instance=params["instance"]

            if "url" in params:
                url=params["url"]
            params_json={}
            if "params" in params:
                params_json=params["params"]
        
        
            #Getting model info, data, and file from local repository
            model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

            #loading data
            if data_file!=None:
                dataframe = pd.read_csv(data_file,header=0)
            else:
                return "The training data file was not provided.",BAD_REQUEST

            ##getting params from info
            model_info=json.load(model_info_file)
            backend = model_info["backend"] 
            target_names=model_info["attributes"]["target_names"]
            features=list(model_info["attributes"]["features"].keys())
            for target in target_names:
                features.remove(target)
            feature=features[0]
            X_train=dataframe.drop(target_names,axis=1).to_numpy()
            X_train=X_train.reshape(tuple(X_train.shape[:1])+tuple(model_info["attributes"]["features"][feature]["shape"]))
            y_train=dataframe[target_names[0]].to_numpy()
            feature=features[0]
        
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
                        return "Could not extract prediction function from model: " + str(e),BAD_REQUEST
            elif url!=None:
                def predict(X):
                    return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
                predic_func=predict
            else:
                return "Either a stored model or a valid URL for the prediction function must be provided.",BAD_REQUEST
            #check univariate
            if(1):
                pass
            else:
                return "This method only supports univariate timeseries datasets.",BAD_REQUEST

            #getting params from request
            distance='dtw' 
            if "distance" in params_json: 
                distance = str(params_json["distance"])
            n=3
            if "n_neighbours" in params_json: 
                n = int(params_json["n_neighbours"])

            #reshaping instance
            instance=np.array(instance)
            instance=instance.reshape(model_info["attributes"]["features"][feature]["shape"])

            #getting prediction for instance
            preds=predic_func(np.expand_dims(instance,axis=0))[0]
            if(len(preds.shape)==1): #proba
                y_pred=np.argmax(np.asarray(preds))
            else:
                y_pred=preds


            #explanation
            df = pd.DataFrame(y_train, columns = ['label'])
            df.index.name = 'index'

            knn = KNeighborsTimeSeries(n_neighbors=n, metric=distance).fit(X_train[list(df[df['label'] == y_pred].index.values)])
            neighbours=knn.kneighbors(instance.reshape(1,-1), return_distance=True)

            df=pd.DataFrame([instance.flatten()] + [X_train[i].flatten() for i in neighbours[1][0]]).transpose()
            df.columns=["Query"]+["Neighbour " +str(i) for i in range(1,len(neighbours[1][0])+1)]

            print(neighbours)
            #plotting
            fig = px.line(df, title="K-Nearest Neighbours")
            fig.update_layout(title_x=0.45,
                xaxis_title="Time", yaxis_title=feature,legend_title=""
            )
            exp=fig.to_html(include_plotlyjs="cdn").replace("\n"," ").replace('"',"'")

            response={"type":"html","explanation":exp}#,"explanation":dict_exp}
            return response
        except:
            return traceback.format_exc(), 500        

    def get(self,id=None):
        base_dict={
        "_method_description": "Find the nearest neighbours of a query based on the training data. This method accepts 3 arguments: " 
                           "the 'id', the 'instance', and the 'params' dictionary (optional) with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
                   "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "instance": "Array containing the values for each time point.",
        "params": { 
                "distance" :{
                    "description":"Distance metric to be used. Defaults to 'dtw'.",
                    "type":"string",
                    "default":'dtw',
                    "range":['dtw', 'softdtw', 'ctw', 'euclidean', 'sqeuclidean'],
                    "required":False
                    },
                "n_neighbours":{
                    "description":"Number of neighbours to be shown. Default is 3.",
                    "type":"int",
                    "default":3,
                    "range":None,
                    "required":False
                    }

                },
        "output_description":{
                "timeseries_nearestneighbours": "Show the nearest neighbours of the query based on the training data."
        },
        "meta":{
                "modelAccess":"Any",
                "supportsBWImage":False,
                "needsTrainingData": True
            }

        }

        if id is not None:
            #Getting model info, data, and file from local repository
            try:
                _, model_info_file, data_file = get_model_files(id,self.model_folder)
            except:
                return base_dict

            dataframe = pd.read_csv(data_file,header=0)
            model_info=json.load(model_info_file)
            target_names=model_info["attributes"]["target_names"]
            features=list(model_info["attributes"]["features"].keys())
            for target in target_names:
                features.remove(target)
            dataframe.drop(target_names,axis=1,inplace=True)


            base_dict["params"]["n_neighbours"]["range"]=[1,dataframe.shape[0]]


            return base_dict
           

        else:
            return base_dict

        
