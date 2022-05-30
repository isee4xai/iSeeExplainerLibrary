from flask_restful import Resource,reqparse
from flask import request
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import h5py
import joblib
import json
import werkzeug
import matplotlib.pyplot as plt
from alibi.explainers import AnchorImage
from saveinfo import save_file_info


class AnchorsImage(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        
        model = args.get("model")
        image = args.get("image")
        params_json = json.loads(args.get("params"))
        backend = params_json["backend"]

        
        if image==None:
            image = np.array(params_json["image"])
        else:
            image = np.asarray(Image.open(image))
        if len(image.shape)<3:
            image = image.reshape(image.shape + (1,))
            plt.gray()

        if backend=="TF1" or backend=="TF2":
            model=h5py.File(model, 'w')
            mlp = tf.keras.models.load_model(model)
            predic_func=mlp.predict
        elif backend=="sklearn":
            mlp = joblib.load(model)
            predic_func =mlp.predict_proba
        elif backend=="PYT":
            mlp = torch.load(model)
            predic_func=mlp.predict
        else:
            mlp = joblib.load(model)
            predic_func=mlp.predict


        segmentation_fn='slic'
        if "segmentation_fn" in params_json:
            segmentation_fn = params_json["segmentation_fn"]

        threshold=0.95
        if "threshold" in params_json:
            threshold= params_json["threshold"]

        explainer = AnchorImage(predic_func, image.shape, segmentation_fn=segmentation_fn)
        explanation = explainer.explain(image,threshold)
       
        fig, axes = plt.subplots(1,1, figsize = (4, 4))
        axes.imshow(explanation.anchor)
        axes.set_title('Predicted Class: {}'.format(explanation.raw["prediction"][0]))
        
        #saving
        upload_folder, filename, getcall = save_file_info(request.path)
        fig.savefig(upload_folder+filename+".png")

        response={"plot_png":getcall+".png","explanation":json.loads(explanation.to_json())}
        return response

    def get(self):
        return {
        "_method_description": "Displays the pixels that are sufficient to the model to justify the predicted class. "
                           "Requires 3 arguments: " 
                           "the 'model' which is a file containing the trained model, the 'params' JSON with the configuration parameters of the method, and optionally a file with the image to be explained. "
                           "These arguments are described below.",

        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models. For models with different backends, also upload "
             "a .pkl, and make sure that the prediction function of the model is called 'predict'. This can be achieved by using a wrapper class.",
        "image": "Image file to be explained. Passing a file is only recommended when the model works with black and white, or color images that are RGB-encoded using integers ranging from 0 to 255. Otherwise, pass the image in the params attribute.",
        "params": { 
                "image": "Matrix representing the image. Ignored if an image file was uploaded.",
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "threshold": "A float from 0 to 1 with the desired precision for the anchor.",
                "segmentation_fn": "(Optional) A string with a segmentation algorithm from the following:'quickshift', 'slic', or 'felzenszwalb'."
                }

        }