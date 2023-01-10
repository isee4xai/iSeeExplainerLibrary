from flask_restful import Resource,reqparse
from flask import request
from PIL import Image
import numpy as np
import tensorflow as tf
import h5py
import json
import werkzeug
import matplotlib.pyplot as plt
from alibi.explainers import IntegratedGradients
from saveinfo import save_file_info
from getmodelfiles import get_model_files



class IntegratedGradientsImage(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument("instance")        
        parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("params")
        args = parser.parse_args()
        
        _id = args.get("id")
        instance = args.get("instance")
        image = args.get("image")
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)

        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  
        output_names=model_info["attributes"]["target_values"][0]


        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
            else:
                raise Exception("This method only supports Tensorflow/Keras models.")
        else:
            raise Exception("This method requires a model file.")

        if instance!=None:
            try:
                image = np.array(json.loads(instance))
            except:
                raise Exception("Could not read instance from JSON.")
        elif image!=None:
             try:
                image = np.asarray(Image.open(image))
             except:
                 raise Exception("Could not load image from file.")
        else:
            raise Exception("Either an image file or a matrix representative of the image must be provided.")

        if image.shape!=tuple(model_info["attributes"]["features"]["image"]["shape"]):
            image = image.reshape(tuple(model_info["attributes"]["features"]["image"]["shape"]))
        if image.shape[-1]==1:
            plt.gray()
        image=image.reshape((1,) + image.shape)

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

        prediction=mlp(image).numpy().argmax(axis=1)
        target_class=int(prediction[0])
        print(target_class)
        if "target_class" in params_json:
             target_class = params_json["target_class"]

        ## Generating explanation
        ig  = IntegratedGradients(mlp,
                                  n_steps=n_steps,
                                  method=method,
                                  internal_batch_size=internal_batch_size)

        explanation = ig.explain(image, target=target_class)
        attrs = explanation.attributions[0]

        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 8))
        fig.set_tight_layout(True)
        cmap_bound = np.abs(attrs).max()

        ax[0].imshow(image.squeeze(), cmap='gray')
        ax[0].set_title("Prediction: " + output_names[prediction[0]],wrap=True)

        # attributions
        attr = attrs[0]
        im = ax[1].imshow(attr.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

        # positive attributions
        attr_pos = attr.clip(0, 1)
        im_pos = ax[2].imshow(attr_pos.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

        # negative attributions
        attr_neg = attr.clip(-1, 0)
        im_neg = ax[3].imshow(attr_neg.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

        ax[1].set_title('Attributions for Class ' + output_names[target_class] + ':', wrap=True)
        ax[2].set_title('Positive attributions',wrap=True);
        ax[3].set_title('Negative attributions',wrap=True);

        for ax in fig.axes:
            ax.axis('off')
    
        fig.colorbar(im);

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        fig.savefig(upload_folder+filename+".png")

        response={"plot_png":getcall+".png"}#,"explanation":json.loads(explanation.to_json())}
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
                "target_class": "(optional) Integer denoting the desired class for the computation of the attributions. Defaults to the predicted class of the instance.",
                "method": "(optional) Method for the integral approximation. The methods available are: 'riemann_left', 'riemann_right', 'riemann_middle', 'riemann_trapezoid', 'gausslegendre'. Defaults to 'gausslegendre'.",
                "n_steps": "(optional) Number of step in the path integral approximation from the baseline to the input instance. Defaults to 5.",
                "internal_batch_size": "(optional) Batch size for the internal batching. Defaults to 100."
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
