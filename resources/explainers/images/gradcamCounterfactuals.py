from flask_restful import Resource
from flask import request
from PIL import Image
import numpy as np
import torch
import json
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from saveinfo import save_file_info
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import base64_to_vector,PIL_to_base64
from utils.img_processing import normalize_img

class GradCamCounterfactuals(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def post(self):
        params = request.json
        if params is None:
            return "The params are missing"

        #check params
        if("id" not in params):
            return "The model id was not specified in the params."
        if("type" not in params):
            return "The instance type was not specified in the params."
        if("instance" not in params):
            return "The instance was not specified in the params."
        
        _id =params["id"]
        instance = params["instance"]
        inst_type=params["type"]
        params_json={}
        if "params" in params:
            params_json=params["params"]


        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  

        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
            else:
                raise Exception("This method only supports Tensorflow/Keras models.")
        else:
            raise Exception("This method requires a model file.")
    
        #converting to vector
        try:
            instance=base64_to_vector(instance)
        except Exception as e:  
            return "Could not convert base64 Image to vector: " + str(e)

        im=instance #Raw format needed for explanation

        #normalizing
        try:
            instance=normalize_img(instance,model_info)
        except Exception as e:
                return  "Could not normalize instance: " + str(e)

        output_names=model_info["attributes"]["features"][model_info["attributes"]["target_names"][0]]["values_raw"]

        #params from request
        target_layer=None
        if "target_layer" in params_json:
            try: 
                target_layer=mlp.get_layer(params_json["target_layer"]).name
            except Exception as e:
                return "The specified target layer " + str(params_json["target_layer"]) + " does not exist: " + str(e)
        else:
            for i in range(1,len(mlp.layers)+1):
                    if "convolutional" in str(type(mlp.layers[-i])):
                        target_layer=mlp.layers[-i].name
                        break
            
        if target_layer is None:
            return "No target layer found."

        if "target_layer_index" in params_json:
            try:
                target_layers=[target_layers[0][int(params_json["target_layer_index"])]]
            except:
                return "The specified index could not be accessed in the target_layer." 

        target_class=None
        if "target_class" in params_json:
            if(params_json["target_class"]!="Highest Prob."):
                target_class = str(params_json["target_class"])

        preds=mlp.predict(instance)[0]

        ind=np.argpartition(preds, -2)[-2:]
        sorted_top=ind[np.argsort(preds[ind])]
        pred_index=sorted_top[0]
        cf_index=sorted_top[1]

        pred_index=None
        if target_class is not None and target_class!="Highest Prob.": 
           cf_index=output_names.index(target_class)


        

        #vector to base 64
        b64Image=PIL_to_base64(superimposed_img)
        response={"type":"image","explanation":b64Image}#,"explanation":json.loads(explanation.to_json())}
        return response

    def get(self,id=None):
        base_dict= {
        "_method_description": "Gradient-weighted Class Activation Mapping (Grad-CAM), uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept."
                           "This method accepts 4 arguments: " 
                           "the 'id', the 'params' dictionary (optional) with the configuration parameters of the method, the 'instance' containing the image that will be explained as a matrix, or the 'image' file instead. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Matrix representing the image to be explained.",
        "image": "Image file to be explained. Ignored if 'instance' was specified in the request. Passing a file is only recommended when the model works with black and white images, or color images that are RGB-encoded using integers ranging from 0 to 255.",
        "params": { 
                "target_layer":{
                    "description":  "Name of target layer to be provided as a string. This is the layer that you want to compute the visualization for."\
                                    " Usually this will be the last convolutional layer in the model. It is also possible to specify internal components of this layer by passing the"\
                                    " target_layer_index parameter in params. To get the target layer, this method executes 'model.<target_layer>[<target_layer_index>]'"\
                                    " Some common examples of these parameters for well-known models:"\
                                    " Resnet18 and 50: model.layer4 -> 'target_layer':'layer4'"\
                                    " VGG, densenet161: model.features[-1] -> 'target_layer':'features', 'target_layer_index':-1"\
                                    " mnasnet1_0: model.layers[-1] -> 'target_layer':'layers', 'target_layer_index':-1",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":True
                    },
                "target_layer_index":{
                    "description":  "Index of the target layer to be accessed. Provide it when you want to focus on a specific component of the target layer."\
                                    " If not provided, the whole layer specified as target when uploading the model will be used.",
                    "type":"int",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "target_class":{
                    "description": "String representing the target class to generate the explanation. If not provided, defaults to the class with the highest predicted probability.",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "aug_smooth": {
                    "description": "Boolean indicating whether to apply augmentation smoothing (defaults to True). This has the effect of better centering the CAM around the objects. However, it increases the run time by a factor of x6.",
                    "type":"boolean",
                    "default": True,
                    "range":[True,False],
                    "required":False
                    }
                },
        "output_description":{
                "saliency_map":"Displays an image that highlights the region that contributes the most to the target class."
            },

        "meta":{
                "modelAccess":"File",
                "supportsBWImage":True,
                "needsTrainingData": False
        }
        }

        if id is not None:
            #Getting model info, data, and file from local repository
            try:
                _, model_info_file, _ = get_model_files(id,self.model_folder)
            except:
                return base_dict

            model_info=json.load(model_info_file)
            target_name=model_info["attributes"]["target_names"][0]


            if model_info["attributes"]["features"][target_name]["data_type"]=="categorical":

                output_names=model_info["attributes"]["features"][target_name]["values_raw"]

                base_dict["params"]["target_class"]["default"]="Highest Pred."
                base_dict["params"]["target_class"]["range"]=["Highest Pred."] + output_names

                return base_dict

            else:
                base_dict["params"].pop("target_class")
                return base_dict

        else:
            return base_dict