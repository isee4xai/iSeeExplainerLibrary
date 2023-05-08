from flask_restful import Resource
from flask import request
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import joblib
import h5py
import json
from scipy import signal
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from statsmodels.nonparametric.smoothers_lowess import lowess
from saveinfo import save_file_info
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.dataframe_processing import split_sequences, denormalize_dataframe
from utils.base64 import PIL_to_base64
import requests



class CBRFox(Resource):

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

        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"] 
        target_names=model_info["attributes"]["target_names"]
        features=model_info["attributes"]["features"]
        feature_names=list(features.keys())
        k=model_info["attributes"]["window_size"]

        time_feature=None
        for feature, info_feature in features.items():
            if(info_feature["data_type"]=="time"):
                time_feature=feature
                break       

        #denormalizing instance
        df_instance=pd.DataFrame(instance,columns=feature_names).drop([time_feature], axis=1, errors='ignore')
        denorm_instance=denormalize_dataframe(df_instance, model_info).to_numpy()
        feature_names.remove(time_feature)
        
        #loading data
        if data_file!=None:
            dataframe = pd.read_csv(data_file,header=0)
        else:
            raise Exception("The training data file was not provided.")

        dataframe.drop([time_feature], axis=1,inplace=True)
        numpy_train=split_sequences(dataframe,k)

        target_columns=[feature_names.index(target) for target in target_names]
        nonOutputColumns=list(set([i for i in range(len(feature_names))]).difference(set(target_columns)))
       
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

        #getting params from request
        smoothnessFactor = .03
        punishedSumFactor = .5
        method_list=["average","max","min","median"]
        explicationMethodResult = 1         #1 average, 2 Max values, 3 min values, 4 median
        if "smoothness_factor" in params_json: 
            smoothnessFactor = float(params_json["smoothness_factor"])
        if "punished_sum_factor" in params_json:
            punishedSumFactor = float(params_json["punished_sum_factor"])   
        if "reference_method" in params_json:
            method_str=str(params_json["reference_method"]).lower()
            if method_str in method_list:
                explicationMethodResult=method_list.index(method_str)+1

        #explanation generation
        windows = np.delete(numpy_train, nonOutputColumns, len(numpy_train.shape)-1)
        targetWindow, windows = windows[-1], windows[:-1]
        windowsLen = len(windows)
        componentsLen = windows.shape[-1]
        windowLen = windows.shape[-2]

        prediction = predic_func(np.array([denorm_instance]))
        actualPrediction=prediction[-1]

        titleColumns = [ feature_names[col] for col in target_columns]

        pearsonCorrelation = np.array(([np.corrcoef(windows[currentWindow,:,currentComponent], targetWindow[:,currentComponent])[0][1]
        for currentWindow in range(len(windows)) for currentComponent in range(componentsLen)])).reshape(-1,componentsLen)

        euclideanDistance = np.array(([np.linalg.norm(targetWindow[:,currentComponent] - windows[currentWindow,:,currentComponent])
        for currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,componentsLen)

        normalizedEuclideanDistance = euclideanDistance / np.amax(euclideanDistance,axis=0)

        normalizedCorrelation = (.5+(pearsonCorrelation-2*normalizedEuclideanDistance+1)/4)

        correlationPerWindow = np.sum(((normalizedCorrelation+punishedSumFactor)**2), axis=1)
        correlationPerWindow /= max(correlationPerWindow)

        smoothedCorrelation = lowess(correlationPerWindow, np.arange(len(correlationPerWindow)), smoothnessFactor)[:,1]

        valleyIndex, peakIndex = signal.argrelextrema(smoothedCorrelation, np.less)[0], signal.argrelextrema(smoothedCorrelation, np.greater)[0]

        concaveSegments = np.split(np.transpose(np.array((np.arange(windowsLen), correlationPerWindow))), valleyIndex)
        convexSegments = np.split(np.transpose(np.array((np.arange(windowsLen), correlationPerWindow))), peakIndex)

        bestWindowsIndex, worstWindowsIndex = list(), list()

        for split in concaveSegments:
            bestWindowsIndex.append(int(split[np.where(split == max(split[:,1]))[0][0],0]))
        for split in convexSegments:
            worstWindowsIndex.append(int(split[np.where(split == min(split[:,1]))[0][0],0]))

        bestDic = {index: correlationPerWindow[index] for index in bestWindowsIndex}
        worstDic = {index: correlationPerWindow[index] for index in worstWindowsIndex}

        bestSorted = sorted(bestDic.items(),reverse=True, key=lambda x:x[1])
        worstSorted = sorted(worstDic.items(), key=lambda x:x[1])

        maxComp,minComp,lims=[],[],[]
        for i in range(componentsLen):
            maxComp.append(int(max(max(a) for a in windows[:,:,i])))
            minComp.append(int(min(min(a) for a in windows[:,:,i])))
            lims.append(range(minComp[i],maxComp[i],int((maxComp[i]-minComp[i])/8)))

        bestMAE,worstMAE = [],[]
        for i in range(len(bestSorted)):
            rawBestMAE=rawWorstMAE=0
            for f in range(componentsLen):
                rawBestMAE+=(windows[bestSorted[i][0]][windowLen-1][f]-minComp[f])/maxComp[f]
                rawWorstMAE+=(windows[worstSorted[i][0]][windowLen-1][f]-minComp[f])/maxComp[f]
            bestMAE.append(rawBestMAE/componentsLen) 
            worstMAE.append(rawWorstMAE/componentsLen)

        def subOptions(op):
            if op==1:
                newCase = np.sum(windows[list(dict(bestSorted).keys())], axis=0)/len(bestSorted)
            elif op==2:
                newCase = np.max(windows[list(dict(bestSorted).keys())], axis=0)
            elif op==3:
                newCase = np.min(windows[list(dict(bestSorted).keys())], axis=0)
            elif op==4:
                newCase = np.median(windows[list(dict(bestSorted).keys())], axis=0)
            return newCase

        cont=np.arange(1,windowLen+1)
        plt.figure(figsize=(12,8))
        newCase = np.zeros((windowLen,componentsLen))
        try:
            newCase = subOptions(explicationMethodResult)
        except:
            print("Unavailable option")
        for f in range(componentsLen):
            plt.subplot(componentsLen,1,f+1)
            plt.title(titleColumns[f])
            plt.plot(cont,targetWindow[:,f], '.-k', label = "Target")
            plt.plot(cont,newCase[:,f], '.-g' ,label= "Data")
            plt.plot(windowLen+1,actualPrediction[f], 'dk', label = "Prediction")
            plt.plot(windowLen+1,newCase[windowLen-1][f], 'dg', label = "Next day")
            plt.grid()
            plt.xticks(range(1,windowLen+2,1))
            plt.yticks(lims[f])
        plt.tight_layout()

        #saving
        img_buf = BytesIO()
        plt.savefig(img_buf,bbox_inches="tight")
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response={"type":"image","explanation":b64Image}#,"explanation":dict_exp}
        return response
        
        response={"plot_png":getcall+".png"}
        return response

    def get(self):
        return {
        "_method_description": "This method applies the Case-Based Reasoning paradigm to provide explanations-by-example, where time series are split into different time-window cases that serve"
                           "as explanation cases for the outcome of the prediction model. It has been designed for domain-expert users -without ML skills- that need to understand and how (future)"
                           "predictions could be dependent of past time series windows. It proposes a novel similarity function which deals with both the morphological similarity and the absolute"
                           "proximity between the time series, together with several reuse strategies to generate the explanation cases. It uses an automatic evaluation approach based on computing"
                           "the error (MAE) between the model prediction for and the actual values in the solution of the explanatory case. Finally, this evaluation method is applied to demonstrate"
                           "the performance of the proposal on the given dataset. This method accepts 3 arguments: " 
                           "the 'id', the 'instance',  and the 'params' dictionary (optional) with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "2-D Array representing a single segment with data for multiple rows containing the feature values of particular time-points.",
        "params": { 
                "smoothness_factor" :{
                    "description":"Float ranging from 0 to 1 for the smoothness factor that will be applied for metric computation. Defaults to 0.03.",
                    "type":"float",
                    "default":0.03,
                    "range":[0,1],
                    "required":False
                    },
                "punished_sum_factor":{
                    "description":"Float ranging from 0 to 1 for the punished sum factor that will be applied for metric computation. Defaults to 0.5.",
                    "type":"float",
                    "default":0.5,
                    "range":[0,1],
                    "required":False
                    }, 
                "reference_method": {
                    "description":"The method to be used for the selection of the general cases. It may be 'average','min','max' and 'median'. Defaults to 'average'.",
                    "type":"string",
                    "default":"average",
                    "range":["average","min","max","median"],
                    "required":False
                    }
                },
        "meta":{
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[]
            }

        }
