from flask_restful import Resource,reqparse
from flask import request
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests



class CBRFox(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder    

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id',required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## loading data
        if data_file!=None:
            numpy_train = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        kwargsData = dict(feature_names=None, categorical_features=None,categorical_names=None, class_names=None)
        if "target_columns" in model_info:
            kwargsData["target_columns"] = model_info["target_columns"]
        else:
            raise Exception("The target columns must be specified when uploading the model.")
        if "feature_names" in model_info:
            kwargsData["feature_names"] = model_info["feature_names"]
        else:
            kwargsData["feature_names"]=["Feature_" + str(i) for i in range(numpy_train.shape[-1])]


        ## getting predict function
        predic_func=None
        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp.predict
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
        else:
            raise Exception("A stored model for the prediction function must be provided.")

  
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
        nonOutputColumns=[i for i in range(numpy_train.shape[-1])]
        for i in kwargsData["target_columns"]:
            nonOutputColumns.remove(i)
    
        windows = np.delete(numpy_train, nonOutputColumns, len(numpy_train.shape)-1)
        targetWindow, windows = windows[-1], windows[:-1]
        windowsLen = len(windows)
        componentsLen = windows.shape[-1]
        windowLen = windows.shape[-2]

        prediction = predic_func(np.array([instance]))
        actualPrediction=prediction[-1]

        titleColumns = [ kwargsData["feature_names"][col] for col in kwargsData["target_columns"]]

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

        ##saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        plt.savefig(upload_folder+filename+".png",bbox_inches='tight')
        
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
                "smoothness_factor" : "(Optional) Float ranging from 0 to 1 for the smoothness factor that will be applied for metric computation. Defaults to 0.3.",
                "punished_sum_factor": "(Optional) Float ranging from 0 to 1 for the punished sum factor that will be applied for metric computation. Defaults to 0.5.",
                "reference_method": "(Optional) The method to be used for the selection of the general cases. It may be 'average','min','max' and 'median'. Defaults to 'average'."
                }

        }
