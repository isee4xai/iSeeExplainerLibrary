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

class GradCam(Resource):

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

        is_tf=False
        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                is_tf=True
            elif backend in ontologyConstants.PYTORCH_URIS:
                mlp = torch.load(model_file)
                mlp.eval()

            else:
                raise Exception("This method only supports Tensorflow/Keras or PyTorch models.")
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

        output_names=None
        try:
            output_names=model_info["attributes"]["features"][model_info["attributes"]["target_names"][0]]["values_raw"]
        except:
            pass
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
            if(params_json["target_class"]!="Highest Pred."):
                target_class = str(params_json["target_class"])

        pred_index=None
        if target_class is not None:
            if(output_names is not None):
                try:
                    pred_index=output_names.index(target_class)
                except:
                    pass

        if(is_tf):
            grad_model = tf.keras.models.Model(
                [mlp.inputs], [mlp.get_layer(target_layer).output, mlp.output]
            )

            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(instance)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            # This is the gradient of the output neuron (top predicted or chosen)
            # with regard to the output feature map of the last conv layer
            grads = tape.gradient(class_channel, last_conv_layer_output)


            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            # then sum all the channels to obtain the heatmap class activation
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()

            # Rescale heatmap to a range 0-255
            heatmap = np.uint8(255 * heatmap)

            # Use jet colormap to colorize heatmap
            jet = cm.get_cmap("jet")

            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]

            # Create an image with RGB colorized heatmap
            jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize(tuple(model_info["attributes"]["features"]["image"]["shape_raw"][:2]))
            jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

            # Superimpose the heatmap on original image
            if len(im.shape)==2:
                im=im.reshape(im.shape+(1,))
            superimposed_img = jet_heatmap * 0.4 + im
            superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

            #vector to base 64
            b64Image=PIL_to_base64(superimposed_img)
            response={"type":"image","explanation":b64Image}#,"explanation":json.loads(explanation.to_json())}
            return response

        else:
            if "target_layer" in params_json:
                if hasattr(mlp,params_json["target_layer"]):
                    target_layers = [getattr(mlp,params_json["target_layer"])]
                else:
                    return "The specified target layer " + str(params_json["target_layer"]) + " does not exist."
            else:
                return "This method requires the name of target layer to be provided as a string. This is the layer that you want to compute the visualization for."\
                    " Usually this will be the last convolutional layer in the model. It is also possible to specify internal components of this layer by passing the"\
                    " target_layer_index parameter in params. To get the target layer, this method executes 'model.<target_layer>[<target_layer_index>]'"\
                    " Some common examples of these parameters for well-known models:"\
                    " Resnet18 and 50: model.layer4 -> 'target_layer':'layer4'"\
                    " VGG, densenet161: model.features[-1] -> 'target_layer':'features', 'target_layer_index':-1"\
                    " mnasnet1_0: model.layers[-1] -> 'target_layer':'layers', 'target_layer_index':-1"

            if "target_layer_index" in params_json:
                try:
                    target_layers=[target_layers[0][int(params_json["target_layer_index"])]]
                except:
                    return "The specified index could not be accessed in the target_layer." 

            target=None
            if "target_class" in params_json:
                target = [ClassifierOutputTarget(int(params_json["target_class"]))]

            aug_smooth=True
            if "aug_smooth" in params_json:
                aug_smooth= bool(params_json["aug_smooth"])
      
            image=None 
            input_tensor=None
            if instance!=None:
                try:
                    input_tensor = torch.tensor(json.loads(instance))
                except:
                    raise Exception("Could not read instance from JSON.")
            elif image!=None:
                try:
                    im = Image.open(image)
                except:
                    raise Exception("Could not load image from file.")
                try:
                    transform = transforms.Compose([transforms.ToTensor()])
                    input_tensor=transform(im.copy())
                except Exception as e:
                    print(e)
                    return "Could not convert image to Tensor."
            else:
                raise Exception("Either an image file or a matrix representative of the image must be provided.")

            input_tensor=input_tensor.reshape((1,)+input_tensor.shape)
            cam  = GradCAM(model=mlp,
                       target_layers=target_layers, use_cuda=torch.cuda.is_available())
            grayscale_cam = cam(input_tensor=input_tensor, targets=target,aug_smooth=aug_smooth,eigen_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            transform = transforms.Compose([transforms.ToPILImage()])
            rgb_img=transform(input_tensor[0])
            cam_image = show_cam_on_image(np.float32(rgb_img) / 255, grayscale_cam, use_rgb=True)
  
            #saving
            upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(upload_folder+filename+'.png')

            response={"plot_png":getcall+".png"}
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