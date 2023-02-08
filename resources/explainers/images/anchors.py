from flask_restful import Resource,reqparse
from flask import request
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import h5py
import joblib
import json
import math
import werkzeug
import matplotlib.pyplot as plt
from alibi.explainers import AnchorImage
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests

BACKENDS=["TF1",
	"TF2",
	"TF",
    "TensorFlow1",
    "TensorFlow2",
    "tensorflow1",
    "tensorflow2"]

class AnchorsImage(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument('instance')        
        parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("url")
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        instance = args.get("instance")
        image = args.get("image")
        url = args.get("url")
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)

        output_names=None
        predic_func=None
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        output_names=model_info["attributes"]["features"][model_info["attributes"]["target_names"][0]]["values_raw"]

        if model_file!=None:
            if backend in BACKENDS:
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp.predict
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
            raise Exception("Either a locally stored model or an URL for the prediction function of the model must be provided.")
                
        if instance!=None:
            try:
                image = np.array(json.loads(instance))
            except:
                raise Exception("Could not read instance from JSON.")
        elif image!=None:
            try:
                im = Image.open(image)
            except:
                raise Exception("Could not load image from file.")
            #cropping
            shape_raw=model_info["attributes"]["features"]["image"]["shape_raw"]
            im=im.crop((math.ceil((im.width-shape_raw[0])/2.0),math.ceil((im.height-shape_raw[1])/2.0),math.ceil((im.width+shape_raw[0])/2.0),math.ceil((im.height+shape_raw[1])/2.0)))
            instance=np.asarray(im)
            #normalizing
            if("min" in model_info["attributes"]["features"]["image"] and "max" in model_info["attributes"]["features"]["image"] and
                "min_raw" in model_info["attributes"]["features"]["image"] and "max_raw" in model_info["attributes"]["features"]["image"]):
                nmin=model_info["attributes"]["features"]["image"]["min"]
                nmax=model_info["attributes"]["features"]["image"]["max"]
                min_raw=model_info["attributes"]["features"]["image"]["min_raw"]
                max_raw=model_info["attributes"]["features"]["image"]["max_raw"]
                try:
                    image=((instance-min_raw) / (max_raw - min_raw)) * (nmax - nmin) + nmin
                except:
                    return "Could not normalize instance."
            elif("mean_raw" in model_info["attributes"]["features"]["image"] and "std_raw" in model_info["attributes"]["features"]["image"]):
                mean=np.array(model_info["attributes"]["features"]["image"]["mean_raw"])
                std=np.array(model_info["attributes"]["features"]["image"]["std_raw"])
                try:
                    image=((instance-mean)/std).astype(np.uint8)
                except:
                    return "Could not normalize instance using mean and std dev."
        else:
            raise Exception("Either an image file or a matrix representative of the image must be provided.")

        if image.shape!=tuple(model_info["attributes"]["features"]["image"]["shape"]):
            image = image.reshape(tuple(model_info["attributes"]["features"]["image"]["shape"]))
        if len(model_info["attributes"]["features"]["image"]["shape_raw"])==2 or model_info["attributes"]["features"]["image"]["shape_raw"][-1]==1:
            plt.gray()

        print(image.shape)

        segmentation_fn='slic'
        if "segmentation_fn" in params_json:
            segmentation_fn = params_json["segmentation_fn"]

        threshold=0.95
        if "threshold" in params_json:
            threshold= float(params_json["threshold"])

        size=(4, 4)
        if "png_height" in params_json and "png_width" in params_json:
            try:
                size=(int(params_json["png_width"])/100.0,int(params_json["png_height"])/100.0)
            except:
                print("Could not convert dimensions for .PNG output file. Using default dimensions.")

        explainer = AnchorImage(predic_func, image.shape, segmentation_fn=segmentation_fn)
        explanation = explainer.explain(image,threshold)
        
        fig, axes = plt.subplots(1,1, figsize = size)
        axes.imshow(explanation.anchor)
        if output_names!=None:
            axes.set_title('Predicted Class: {}'.format(output_names[explanation.raw["prediction"][0]]))
        else:
            axes.set_title('Predicted Class: {}'.format(explanation.raw["prediction"][0]))
        
        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        fig.savefig(upload_folder+filename+".png")

        response={"plot_png":getcall+".png"}#,"explanation":json.loads(explanation.to_json())}
        return response

    def get(self):
        return {
        "_method_description": "Uses anchors to find the groups of pixels that are sufficient for the model to justify the predicted class."
                           "This method accepts 5 arguments: " 
                           "the 'id', the 'url' (optional),  the 'params' dictionary (optional) with the configuration parameters of the method, the 'instance' containing the image that will be explained as a matrix, or the 'image' file instead. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (images represented as arrays). It must return N outputs (predictions for each image).",
        "instance": "Matrix representing the image to be explained.",
        "image": "Image file to be explained. Ignored if 'instance' was specified in the request. Passing a file is only recommended when the model works with black and white images, or color images that are RGB-encoded using integers ranging from 0 to 255.",
        "params": { 
                "threshold": "(Optional) A float from 0 to 1 with the desired precision for the anchor.",
                "segmentation_fn": "(Optional) A string with an image segmentation algorithm from the following:'quickshift', 'slic', or 'felzenszwalb'.",
                "png_width":   "(optional) width (in pixels) of the png image containing the explanation.",
                "png_height": "(optional) height (in pixels) of the png image containing the explanation."
                },
        "output_description":{
                "anchor_image":"Displays the pixels that are sufficient for the model to justify the predicted class."
            },

        "meta":{
                "supportsAPI":True,
                "supportsB&WImage":True,
                "needsData": False,
                "requiresAttributes":[]

            }

        }