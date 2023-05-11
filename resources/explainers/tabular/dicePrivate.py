from flask_restful import Resource
import tensorflow as tf
import h5py
import json
import dice_ml
import pandas as pd
from getmodelfiles import get_model_files
from flask import request
from utils import ontologyConstants
from utils.dataframe_processing import denormalize_dataframe
from utils.dataframe_processing import normalize_dict

class DicePrivate(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
    
    def post(self):
        params = request.json
        if params is None:
            return "The json body is missing."
        
        #Check params
        if("id" not in params):
            return "The model id was not specified in the params."
        if("type" not in params):
            return "The instance type was not specified in the params."
        if("instance" not in params):
            return "The instance was not specified in the params."

        _id =params["id"]
        if("type"  in params):
            inst_type=params["type"]
        instance=params["instance"]
        params_json={}
        if "params" in params:
            params_json=params["params"]
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"] 
        outcome_name=model_info["attributes"]["target_names"][0]

        features=model_info["attributes"]["features"]
        feat=features.copy()
        for k,v in feat.items():
            if(v["data_type"]=="numerical"):
                if("min" in v and "max" in v):
                    feat.update({k:[v["min"],v["max"]]})
                else:
                    return "This method needs the minimum and maximum values of each feature. Please include this information in the configuration file."
            elif(v["data_type"]=="categorical"):
                feat.update({k:[str(x) for x in v["values_raw"]]})
        feat.pop(outcome_name)

        #normalize instance
        norm_instance=normalize_dict(instance,model_info)
      
        ## loading model
        model=None
        back=None
        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                back="TF2"
                model=h5py.File(model_file, 'w')
                model = tf.keras.models.load_model(model)
                if(backend==ontologyConstants.TENSORFLOW_URIS[0]):
                    back="TF1"
            else:
                return "This method only supports TensorFlow models."
        else:
            return "The model file was not uploaded."

        kwargsData = dict(features=feat, outcome_name=outcome_name)

        desired_class=0
        if(len(features[outcome_name]["values_raw"])==2): #binary classification
            desired_class="opposite"
        kwargsData2 = dict(desired_class=desired_class,total_CFs=3)
        if "num_cfs" in params_json:
           kwargsData2["total_CFs"] = int(params_json["num_cfs"])
        if "desired_class" in params_json:
           kwargsData2["desired_class"] = params_json["desired_class"] if params_json["desired_class"]=="opposite" else int(params_json["desired_class"])
        if "features_to_vary" in params_json:
           kwargsData2["features_to_vary"] = params_json["features_to_vary"] if params_json["features_to_vary"]=="all" else json.loads(params_json["features_to_vary"])

        # Create data
        d = dice_ml.Data(**{k: v for k, v in kwargsData.items() if v is not None})
  
        # Create model
        m = dice_ml.Model(model=model, backend=back)

        # Create CFs generator using random
        method="random"
        if "method" in params_json:
           method = params_json["method"]
        exp = dice_ml.Dice(d, m, method=method)

       
        # Generate counterfactuals
        e1 = exp.generate_counterfactuals(pd.DataFrame(norm_instance,index=[0]), **{k: v for k, v in kwargsData2.items() if v is not None})

        #saving
        str_html=''
        i=1
        for cf in e1.cf_examples_list:
            cf.final_cfs_df=denormalize_dataframe(cf.final_cfs_df,model_info)
            if cf.final_cfs_df is None:
                cfs = "<h3>No counterfactuals were found for this instance. Try to vary different features.</h3>"    
           
            else:
                cfs=cf.final_cfs_df.to_html()

            str_html =  str_html + '<h2>Instance ' + str(i) + '</h2>' + denormalize_dataframe(cf.test_instance_df,model_info).to_html() + '<h2>Counterfactuals</h2>'+ cfs + '<br><br><hr><br>'
            i=i+1

        #file = open(upload_folder+filename+".html", "w")
        #file.write(str_html)
        #file.close()

        #hti = Html2Image()
        #hti.output_path= upload_folder
        #if "png_height" in params_json and "png_width" in params_json:
        #    size=(int(params_json["png_width"]),int(params_json["png_height"]))
        #    hti.screenshot(html_str=str_html, save_as=filename+".png",size=size)
        #else:
        #    hti.screenshot(html_str=str_html, save_as=filename+".png")
        
        response={"type":"html","explanation":str_html}
        return response

    def get(self,id=None):
        return {
        "_method_description": "Diverse Counterfactual Explanations (DiCE)  private method generates counterfactuals without training data. However, it requires the format and ranges of the data to be specified when uploading the model. This method is currently supported for TensorFlow models only.  Accepts 3 arguments: " 
                           "the 'id' string, the 'instance', and the 'params' dictionary (optional) containing the configuration parameters of the explainer."
                           " These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance without including the target class.",
        "params": { 

                "desired_class": {
                    "description": "Integer representing the index of the desired counterfactual class. Defaults to class 0 for multiclass problems and to opposite class for binary class problems. You may also use the string 'opposite' for binary classification",
                    "type":"int",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "features_to_vary": {
                    "description": "List of strings representing the feature names to vary. Defaults to all features.",
                    "type":"array",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "num_cfs": {
                    "description": "Number of counterfactuals to be generated for each instance.",
                    "type":"int",
                    "default": 3,
                    "range":None,
                    "required":False
                    },
                "method": {
                    "description": "The method used for counterfactual generation. The supported methods for private data are: 'random' (random sampling) and 'genetic' (genetic algorithms). Defaults to 'random'.",
                    "type":"string",
                    "default": "random",
                    "range":["random","genetic"],
                    "required":False
                    }
        },
        "output_description":{
                "html_table": "An html page containing a table with the original instance compared against a table with the generated couterfactuals."
               },
        "meta":{
                "supportsAPI":False,
                "needsData": False,
                "needsMin&Max":True,
                "requiresAttributes":[]
            }
        }
