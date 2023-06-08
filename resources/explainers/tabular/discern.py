from flask_restful import Resource
import json
import pandas as pd
import numpy as np
from getmodelfiles import get_model_files
import joblib
import h5py
from html2image import Html2Image
from flask import request
from discern import discern_tabular
import tensorflow as tf
from utils import ontologyConstants
from utils.dataframe_processing import normalize_dataframe, denormalize_dataframe

class DisCERN(Resource):

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

        return self.explain(_id, instance, params_json)
    
    def explain(self, model_id, instance, params_json):
        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(model_id, self.model_folder)


        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) 
        else:
            raise Exception("The training data file was not provided.")

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        features=model_info["attributes"]["features"]
        target_name=model_info["attributes"]["target_names"][0]
        output_names=model_info["attributes"]["features"][target_name]["values_raw"]
        outcome_name=target_name
        feature_names=list(dataframe.columns)
        feature_names.remove(outcome_name)

        categorical_features=[]
        for feature in feature_names:
            if features[feature]["data_type"]=="categorical":
                categorical_features.append(dataframe.columns.get_loc(feature))
      
        ## loading model
        model=None
        if model_file!=None:
            if backend in ontologyConstants.TENSORFLOW_URIS:
                model=h5py.File(model_file, 'w')
                model = tf.keras.models.load_model(model)
            elif backend in ontologyConstants.SKLEARN_URIS:
                model = joblib.load(model_file)
            else:
                return "This method currently supports Tensoflow and scikit-learn classification models only."
        else:
            raise Exception("The model file was not uploaded.")
        

        #getting params from request
        desired_class=0
        if(len(output_names)==2): #binary classification
            desired_class="opposite"
        feature_attribution_method='LIME'
        imm_features = []
        if "desired_class" in params_json:
            if params_json["desired_class"]!="opposite":
                if params_json["desired_class"] in output_names:
                    desired_class = output_names.index(params_json["desired_class"])
        if "feature_attribution_method" in params_json: 
            feature_attribution_method = params_json["feature_attribution_method"]  
        if "immutable_features" in params_json and params_json["immutable_features"]: 
            imm_features = [dataframe.columns.get_loc(c) for c in params_json["immutable_features"] if c in dataframe]

        ## init discern
        discern_obj = discern_tabular.DisCERNTabular(model, feature_attribution_method)    
        dataframe_features = dataframe.drop([target_name], axis=1, inplace=False).values
        dataframe_labels = dataframe.loc[:,target_name].values
        target_values = dataframe[outcome_name].unique().tolist()
        discern_obj.init_data(dataframe_features, dataframe_labels, feature_names, target_values, **{'cat_feature_indices':categorical_features, 'immutable_feature_indices':imm_features})


        #normalize instance
        df_inst=pd.DataFrame([instance.values()],columns=instance.keys())
        if target_name in df_inst.columns:
            df_inst.drop([target_name], axis=1, inplace=True)
        df_inst=df_inst[feature_names]
        norm_instance=normalize_dataframe(df_inst,model_info).to_numpy()

        test_label = None 
        if backend in ontologyConstants.TENSORFLOW_URIS:
            test_label = model.predict(norm_instance).argmax(axis=-1)[0]
        elif backend in ontologyConstants.SKLEARN_URIS:
            test_label = model.predict(norm_instance)[0]

        try:
            cf, cf_label, s, p = discern_obj.find_cf(norm_instance[0], test_label, desired_class)
        except Exception as e:
            print(e)
            return {"type":"text", "explanation":"Counterfactual not found."}
            
        norm_instance=np.append(norm_instance,test_label)
        cf=np.append(cf,cf_label)
        feature_names.append(outcome_name)

        result_df = pd.DataFrame(np.array([norm_instance, cf]), columns=feature_names)
        result_df_norm=denormalize_dataframe(result_df,model_info)

        #saving
        str_html= result_df_norm.to_html()+'<br>'

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
       
        
        response={"type":"html","explanation":str_html.replace("\n"," ")}
        return response
    
    def get(self,id=None):
        
        base_dict={
        "_method_description": "Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods (DisCERN) generates counterfactuals for scikit-learn-based models. Requires 3 arguments: " 
                           "the 'id' string, the 'instance' to be explained, and the 'params' object containing the configuration parameters of the explainer."
                           " These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array of feature values of the instance to be explained.",
        "params": {
                "desired_class": {
                    "description": "String representing the name of the desired counterfactual class, or 'opposite' in the case of binary classification. Defaults to 'opposite'.",
                    "type":"string",
                    "default": None,
                    "range":None,
                    "required":False
                    },
                "feature_attribution_method":{
                    "description": "Feature attribution method used for obtaining feature weights; currently supports LIME, SHAP and Integrated Gradients. Defaults to LIME.",
                    "type":"string",
                    "default": "LIME",
                    "range":["LIME","SHAP","IntG"],
                    "required":False
                    },
                "immutable_features": {
                    "description": "Array of feature names that are immutable. The counterfactual will not recommend to change these features. Default is an empty array",
                    "type":"array",
                    "default": [],
                    "range":None,
                    "required":False
                    },
                },

        "output_description":{
                "html_table": "An html page containing a table with the generated couterfactuals."
               },
        "meta":{
                "modelAccess":"File",
                "supportsBWImage":False,
                "needsTrainingData": True
        }
    }

        if id is not None:
            #Getting model info, data, and file from local repository
            try:
                _, model_info_file, data_file = get_model_files(id,self.model_folder)
            except:
                return base_dict


            dataframe = joblib.load(data_file)
            model_info=json.load(model_info_file)
            target_name=model_info["attributes"]["target_names"][0]
            output_names=model_info["attributes"]["features"][target_name]["values_raw"]
            feature_names=list(dataframe.columns)
            feature_names.remove(target_name)

            if(len(output_names)==2): #binary classification
                base_dict["params"]["desired_class"]["range"]=["opposite"] + output_names
                base_dict["params"]["desired_class"]["default"]="opposite"
            else:
                base_dict["params"]["desired_class"]["range"]=output_names
                base_dict["params"]["desired_class"]["default"]=output_names[0]

            base_dict["params"]["immutable_features"]["range"]=feature_names
        
            return base_dict
        else:
            return base_dict
