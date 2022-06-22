from flask_restful import Resource,reqparse
from flask import request
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import h5py
import joblib
import json
import werkzeug
import matplotlib.pyplot as plt
from lime import lime_image
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests

class LimeImage(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id",required=True)
        parser.add_argument("url")
        parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        image = args.get("image")
        params_json = json.loads(args.get("params"))

        output_names=None
        predic_func=None
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id)

        ## params from info
        model_info=json.loads(json.load(model_info_file))
        backend = model_info["backend"]  ##error handling?
        if "output_names" in model_info:
            output_names=model_info["output_names"]

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
            raise "Either an ID for a locally stored model or an URL for the prediction function of the model must be provided."
        
        
        if image==None:
            try:
                image = np.array(params_json["image"])
            except:
                raise "Either an image file or a matrix representative of the image must be provided."
        else:
            image = np.asarray(Image.open(image))

        
        kwargsData = dict(top_labels=3,segmentation_fn=None)
        if "top_classes" in params_json:
            kwargsData["top_labels"] = params_json["top_classes"]   #top labels
        if "segmentation_fn" in params_json:
            kwargsData["segmentation_fn"] = SegmentationAlgorithm(params_json["segmentation_fn"])
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image, classifier_fn=predic_func,**{k: v for k, v in kwargsData.items() if v is not None})

        fig, axes = plt.subplots(1,len(explanation.top_labels), figsize = (12, 12))
        i=0
        for cat in explanation.top_labels:
            temp, mask = explanation.get_image_and_mask(cat, positive_only=False,hide_rest=False)
            axes[i].imshow(mark_boundaries(temp, mask))
            if output_names != None:
                cat = output_names[cat]
            axes[i].set_title('Positive/Negative Regions for {}'.format(cat))
            i=i+1
         
        ##formatting json explanation
        dict_exp={}
        for key in explanation.local_exp:
            dict_exp[int(key)]=[[float(y) for y in x ] for x in explanation.local_exp[key]]
        
        #saving
        upload_folder, filename, getcall = save_file_info(request.path)
        fig.savefig(upload_folder+filename+".png")

        response={"plot_png":getcall+".png","explanation":dict_exp}
        return response

    def get(self):
        return {
        "_method_description": "Displays the group of pixels that contribute positively or negatively to the prediction of the image class."
                           "This method accepts 4 arguments: " 
                           "the 'id', the 'url',  the 'params' JSON with the configuration parameters of the method, and optionally the 'image' that will be explained. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally. If provided, then 'url' is ignored.",
        "url": "External URL of the prediction function. This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (images represented as arrays). It must return N outputs (predictions for each image).",
        "image": "Image file to be explained. Passing a file is only recommended when the model works with black and white images, or color images that are RGB-encoded using integers ranging from 0 to 255. Otherwise, pass the image in the params attribute.",
        "params": { 
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probablity to be explained.",
                "segmentation_fn": "(Optional) A string with a segmentation algorithm to be used from the following:'quickshift', 'slic', or 'felzenszwalb'"

                }

        }