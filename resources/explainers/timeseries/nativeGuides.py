from flask_restful import Resource
from flask import request
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import h5py
import json
import joblib
import requests
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.neighbors import KNeighborsTimeSeries
from io import BytesIO
from PIL import Image
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import PIL_to_base64




class NativeGuides(Resource):

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
        
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        #loading data
        if data_file!=None:
            dataframe = pd.read_csv(data_file,header=0)
        else:
            raise Exception("The training data file was not provided.")

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

        #check univariate
        if(1):
            pass
        else:
            return "This method only supports univariate timeseries datasets."

        #check binary class
        if(1):
            pass
        else:
            return "This method only supports binary classification tasks."

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


        #finding the nearest unlike neighbour. NB will need to account for regularization
        def native_guide_retrieval(query, predicted_label, distance, n_neighbors):
    
            df = pd.DataFrame(y_train, columns = ['label'])
            df.index.name = 'index'
            #df[df['label'] == 1].index.values, df[df['label'] != 1].index.values
    
            ts_length = X_train.shape[1]
    
            knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)

            print(y_pred)
            print(df['label']!= predicted_label)
            knn.fit(X_train[list(df[df['label'] != predicted_label].index.values)])
    
            dist,ind = knn.kneighbors(query.reshape(1,ts_length), return_distance=True)
            return dist[0], df[df['label'] != predicted_label].index[ind[0][:]]


        #getting params from request
        distance='dtw' 
        if "distance" in params_json: 
            distance = str(params_json["distance"])

        #reshaping instance
        instance=np.array(instance)
        instance=instance.reshape(model_info["attributes"]["features"][feature]["shape"])


        preds=predic_func(np.expand_dims(instance,axis=0))[0]
        if(len(preds.shape)==1): #proba
            y_pred=np.argmax(np.asarray(preds))
        else:
            y_pred=preds

        nun_index=native_guide_retrieval(instance,y_pred,distance,1)[1][0]

        #explanation
        query = instance
        beta = 0
        insample_cf = X_train[nun_index]
        target = int(not y_pred)
        pred_treshold = 0.5

        generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))

        prob_target = predic_func(generated_cf.reshape(1,-1))[0][target]

        # TODO: include timeout
        while prob_target < pred_treshold:
            beta +=0.01 
            generated_cf = generated_cf = dtw_barycenter_averaging([query, insample_cf], weights=np.array([(1-beta), beta]))
            prob_target = predic_func(generated_cf.reshape(1,-1))[0][target]
        
        #plotting
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0'  # very light grey
        #for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        #    plt.rcParams[param] = '#212946'  # bluish dark grey
        colors = ['#08F7FE',  # teal/cyan
            '#FE53BB', # pink
            '#F5D300',  # yellow
            '#00ff41',  # matrix green
        ]
        df = pd.DataFrame({'Predicted: ' +str(model_info["attributes"]["features"][target_names[0]]["values_raw"][y_pred]): list(instance.flatten()),
                           'Counterfactual: ' +str(model_info["attributes"]["features"][target_names[0]]["values_raw"][int(not y_pred)]): list(generated_cf.flatten())})
        fig, ax = plt.subplots(figsize=(10,5))
        df.plot(marker='.', color=colors, ax=ax)
        # Redraw the data with low alpha and slighty increased linewidth:
        n_shades = 10
        diff_linewidth = 1.05
        alpha_value = 0.3 / n_shades
        for n in range(1, n_shades+1):
            df.plot(marker='.',
                    linewidth=2+(diff_linewidth*n),
                    alpha=alpha_value,
                    legend=False,
                    ax=ax,
                    color=colors)

        ax.grid(color='#2A3459')
        plt.xlabel('Time', fontweight = 'bold', fontsize='xx-large')
        plt.ylabel(feature, fontweight = 'bold', fontsize='xx-large')

        #saving
        img_buf = BytesIO()
        plt.savefig(img_buf,bbox_inches="tight")
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)
        plt.close()

        response={"type":"image","explanation":b64Image}#,"explanation":dict_exp}
        return response
        

    def get(self,id=None):
        base_dict={
        "_method_description": "LIMESegment is a novel time series explanation framework which outperforms existing adaptations of LIME to time series on a variety of classification tasks. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url', and the 'params' dictionary (optional) with the configuration parameters of the method. "
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
                    "range":['dtw','euclidean'],
                    "required":False
                    },
                "window_size":{
                    "description":"Window size to be used by the segmentation algorithm. Default is T/5 (T = series length).",
                    "type":"int",
                    "default":None,
                    "range":None,
                    "required":False
                    },
                "change_points":{
                    "description":"Number of change points to be determined by the segmentation algorithm. Default is 3.",
                    "type":"int",
                    "default":3,
                    "range":None,
                    "required":False
                    }
                },
        "output_description":{
                "timeseries_attributions": "Show the attributions of the individual segments of the timeseries to the positive class."
        },
        "meta":{
                "modelAccess":"Any",
                "supportsBWImage":False,
                "needsTrainingData": False
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
            feature=features[0]
            tslen=len(dataframe.columns)


            base_dict["params"]["window_size"]["default"]=max(3,int(tslen/5))
            base_dict["params"]["window_size"]["range"]=(3,int(tslen/2))

            base_dict["params"]["change_points"]["range"]=[1,tslen]

            return base_dict
           

        else:
            return base_dict

        
