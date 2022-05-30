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
from alibi.explainers import Counterfactual
from saveinfo import save_file_info


class CounterfactualsImage(Resource):
    
    def post(self):
        tf.compat.v1.disable_eager_execution()
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
        image=image.reshape((1,) + image.shape)
        if backend=="TF1" or backend=="TF2":
            model=h5py.File(model, 'w')
            mlp = tf.keras.models.load_model(model)
            predic_func=mlp
        elif backend=="sklearn":
            mlp = joblib.load(model)
            predic_func =mlp.predict_proba
        elif backend=="PYT":
            mlp = torch.load(model)
            predic_func=mlp.predict
        else:
            mlp = joblib.load(model)
            predic_func=mlp.predict


        kwargsData = dict(target_proba=None,target_class='other')
        if "target_proba" in params_json:
             kwargsData["target_proba"] = params_json["target_proba"]
        if "target_class" in params_json:
             kwargsData["target_class"] = params_json["target_class"]

        cf = Counterfactual(predic_func, shape=image.shape, **{k: v for k, v in kwargsData.items() if v is not None})
        explanation = cf.explain(image)

        pred_class = explanation.cf['class']
        proba = explanation.cf['proba'][0][pred_class]       

        fig, axes = plt.subplots(1,1, figsize = (4, 4))
        axes.imshow(explanation.cf['X'][0])
        axes.set_title('Original Class: {}\nCounterfactual Class: {} Probability {:.3f}'.format(explanation.orig_class,pred_class,proba))  

        #saving
        upload_folder, filename, getcall = save_file_info(request.path)
        fig.savefig(upload_folder+filename+".png")

        response={"plot_png":getcall+".png","explanation":json.loads(explanation.to_json())}
        return response

    def get(self):
        return {
        "_method_description": "Displays an image that is as similar as possible to the original but with a different prediction. "
                           "Requires 3 arguments: " 
                           "the 'model' which is a file containing the trained model, the 'params' JSON with the configuration parameters of the method, and optionally a file with the image to be explained. "
                           "These arguments are described below.",

        "model": "The trained prediction model given as a file. The extension must match the backend being used i.e.  a .pkl " 
             "file for Scikit-learn (use Joblib library), .pt for PyTorch or .h5 for TensorFlow models. For models with different backends, also upload "
             "a .pkl, and make sure that the prediction function of the model is called 'predict'. This can be achieved by using a wrapper class.",
        "params": { 
                "image": "Matrix representing the image. Ignored if an image file was uploaded.",
                "backend": "A string containing the backend of the prediction model. The supported values are: 'sklearn' (Scikit-learn), 'TF1' "
                "(TensorFlow 1.0), 'TF2' (TensorFlow 2.0), 'PYT' (PyTorch).",
                "target_class": "A string containing 'other' or 'same', or an integer denoting the desired class for the counterfactual instance.",
                "target_proba": "Float from 0 to 1 representing the target probability for the counterfactual generated."
                }

        }