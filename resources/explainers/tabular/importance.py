from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import joblib
import h5py
import json
import dalex as dx
from flask import request
from html2image import Html2Image
from saveinfo import save_file_info
from getmodelfiles import get_model_files


class Importance(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder 
        
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument("params")

        #parsing args
        args = parser.parse_args()
        _id = args.get("id")
        params=args.get("params")
        params_json= {}
        if params!=None:
            params_json = json.loads(params)
       
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        model_task = model_info["model_task"]  ##error handling?

        ## loading model
        if backend=="TF1" or backend=="TF2":
            model_h5=h5py.File(model_file, 'w')
            model = tf.keras.models.load_model(model_h5)
        elif backend=="PYT":
            model = torch.load(model_file)
        else:
            model = joblib.load(model_file)
        
        ## loading data
        dataframe = joblib.load(data_file)

        ## params from the request
        kwargsData = dict()
        if "variables" in params_json:
            kwargsData["variables"] = params_json["variables"]
       
        explainer = dx.Explainer(model, dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False), dataframe.iloc[:,-1:],model_type=model_task)
        parts = explainer.model_parts(**{k: v for k, v in kwargsData.items()})

        fig=parts.plot(show=False)
        
        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        fig.write_html(upload_folder+filename+'.html')
        hti = Html2Image()
        hti.output_path= upload_folder
        hti.screenshot(html_file=upload_folder+filename+'.html', save_as=filename+".png") 

        response={"plot_html":getcall+'.html',"plot_png":getcall+'.png', "explanation":json.loads(parts.result.to_json())}
        return response


    def get(self):
        return {
        "_method_description": "This method measures the increase in the prediction error of the model after the feature's values are randomly permuted. " 
                                "A feature is considered important if the error of the model increases significantly when permuting it. Accepts 2 arguments: " 
                           "the 'id' string, and the 'params' object (optional) containing the configuration parameters of the explainer."
                           " These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "variables": "(Optional) Array of strings with the names of the features for which the importance will be calculated. Defaults to all features.",
                },

        "params_example":{
                    "variables": [
                                    "1. Most of the time I have difficulty concentrating on simple tasks",
                                    "2. I don't feel like doing my daily duties",
                                    "3. My friends or family have told me that I look different",
                                    "4. When I think about the future it is difficult for me to imagine it clearly",
                                    "5. People around me often ask me how I feel",
                                    "6. I consider that my life is full of good things",
                                    "7. My hobbies are still important to me"
                                  ]
               }
        }
    