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
from skimage.color import gray2rgb, rgb2gray, label2rgb
from lime import lime_image
from saveinfo import save_file_info


class LimeImage(Resource):
    
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


        explainer = lime_image.LimeImageExplainer()
        kwargsData = dict(top_labels=3,segmentation_fn=None)

        if "top_classes" in params_json:
            kwargsData["top_labels"] = params_json["top_classes"]   #top labels
        if "segmentation_fn" in params_json:
             kwargsData["segmentation_fn"] = SegmentationAlgorithm(params_json["segmentation_fn"])

        explanation = explainer.explain_instance(image, classifier_fn=predic_func,**{k: v for k, v in kwargsData.items() if v is not None})

        fig, axes = plt.subplots(1,len(explanation.top_labels), figsize = (12, 12))
        i=0
        for cat in explanation.top_labels:
            temp, mask = explanation.get_image_and_mask(cat, positive_only=False,hide_rest=False)
            axes[i].imshow(mark_boundaries(temp, mask))
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
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probablity to be explained.",
                "segmentation_fn": "(Optional) A string with a segmentation algorithm to be used from the following:'quickshift', 'slic', or 'felzenszwalb'"

                }

        }