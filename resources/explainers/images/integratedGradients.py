from flask_restful import Resource,reqparse
from flask import request
from PIL import Image
import numpy as np
import tensorflow as tf
import h5py
import json
import matplotlib.pyplot as plt
from alibi.explainers import IntegratedGradients
from io import BytesIO
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.base64 import base64_to_vector,PIL_to_base64
from utils.img_processing import normalize_img



class IntegratedGradientsImage(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("params", required=True)
        args = parser.parse_args()

        #Check params
        params_str = args.get('params')
        if params_str is None:
            return "The params were not specified."
        params={}
        try:
            params = json.loads(params_str)
        except Exception as e:
            return "Could not convert params to JSON: " + str(e)

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
        if "ex_params" in params:
            params_json=params["ex_params"]

        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  
        output_names=None
        try:
            output_names=model_info["attributes"]["features"][model_info["attributes"]["target_names"][0]]["values_raw"]
        except:
            pass

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

        if len(model_info["attributes"]["features"]["image"]["shape_raw"])==2 or model_info["attributes"]["features"]["image"]["shape_raw"][-1]==1:
            plt.gray()

        ## params from request
        n_steps = 50
        if "n_steps" in params_json:
            n_steps = params_json["n_steps"]

        method = "gausslegendre"
        if "method" in params_json:
            method = params_json["method"]

        internal_batch_size=100
        if "internal_batch_size" in params_json:
            internal_batch_size = params_json["internal_batch_size"]

        prediction=mlp(instance)[0].numpy()
        target_class=int(prediction.argmax())

        is_class=True
        if(prediction.shape[-1]==1): ## it's regression
            is_class=False

        if(is_class):
            if "target_class" in params_json:
                    target_class = params_json["target_class"]

        size=(12, 4)
        if "png_height" in params_json and "png_width" in params_json:
            try:
                size=(int(params_json["png_width"])/100.0,int(params_json["png_height"])/100.0)
            except:
                print("Could not convert dimensions for .PNG output file. Using default dimensions.")

        ## Generating explanation
        ig  = IntegratedGradients(mlp,
                                  n_steps=n_steps,
                                  method=method,
                                  internal_batch_size=internal_batch_size)

        explanation = ig.explain(instance, target=target_class)
        attrs = explanation.attributions[0]

        fig, (a0,a1,a2,a3,a4) = plt.subplots(nrows=1, ncols=5, figsize=size,gridspec_kw={'width_ratios':[3,3,3,3,1]})
        cmap_bound = np.abs(attrs).max()

        a0.imshow(im)
        a0.set_title("Original Image")

        # attributions
        attr = attrs[0]
        im = a1.imshow(attr.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

        # positive attributions
        attr_pos = attr.clip(0, 1)
        im_pos = a2.imshow(attr_pos.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

        # negative attributions
        attr_neg = attr.clip(-1, 0)
        im_neg = a3.imshow(attr_neg.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

        if(is_class):
            a1.set_title('Attributions for Class: ' + output_names[target_class])
        else:
           a1.set_title("Attributions for Pred: " + str(np.squeeze(prediction).round(4)))
        a2.set_title('Positive attributions');
        a3.set_title('Negative attributions');

        for ax in fig.axes:
            ax.axis('off')
   
        fig.colorbar(im)
        fig.tight_layout()

        #saving
        img_buf = BytesIO()
        fig.savefig(img_buf,bbox_inches='tight',pad_inches = 0)
        im = Image.open(img_buf)
        b64Image=PIL_to_base64(im)

        response={"type":"image","explanation":b64Image}#,"explanation":json.loads(explanation.to_json())}
        return response

    def get(self):
        return {
        "_method_description": "Defines an attribution value for each pixel in the image provided based on the Integration Gradients method. It only works with Tensorflow/Keras models."
                            "This method accepts 4 arguments: " 
                           "the 'id', the 'params' dictionary (optional) with the configuration parameters of the method, the 'instance' containing the image that will be explained as a matrix, or the 'image' file that can be passed instead of the instance. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally. If provided, then 'url' is ignored.",
        "instance": "Matrix representing the image to be explained.",
        "image": "Image file to be explained. Ignored if 'instance' was specified in the request. Passing a file is only recommended when the model works with black and white images, or color images that are RGB-encoded using integers ranging from 0 to 255.",
        "params": { 
                "target_class": "(optional) Integer denoting the desired class for the computation of the attributions. Ignore for regression models. Defaults to the predicted class of the instance.",
                "method": "(optional) Method for the integral approximation. The methods available are: 'riemann_left', 'riemann_right', 'riemann_middle', 'riemann_trapezoid', 'gausslegendre'. Defaults to 'gausslegendre'.",
                "n_steps": "(optional) Number of step in the path integral approximation from the baseline to the input instance. Defaults to 5.",
                "internal_batch_size": "(optional) Batch size for the internal batching. Defaults to 100.",
                "png_width":  "(optional) width (in pixels) of the png image containing the explanation.",
                "png_height": "(optional) height (in pixels) of the png image containing the explanation."
                },
        "output_description":{
                "attribution_plot":"Subplot with four columns. The first column shows the original image and its prediction. The second column shows the values of the attributions for the target class. The third column shows the positive valued attributions. The fourth column shows the negative valued attributions."
            },

        "meta":{
                "supportsAPI":False,
                "supportsB&WImage":True,
                "needsData": False,
                "requiresAttributes":[]
            }

        }
