from flask_restful import Resource
from flask import request
import json
import pandas as pd
import numpy as np

class AIModelPerformance(Resource):

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
        if("usecase" not in params):
            return "The usecase was not specified in the params."

        params_json={}
        if "params" in params:
            params_json=params["params"]

        return self.explain(params["usecase"], params_json)

    def explain(self, usecase, params_json):
        selected_metrics = [f.lower() for f in params_json["selected_metrics"]] if "selected_metrics" in params_json else None
        
        # get case structure from the request
        case_structure = usecase
        #get assessment details from the first case
        assessments = case_structure[0]["http://www.w3id.org/iSeeOnto/explanationexperience#hasDescription"]["http://www.w3id.org/iSeeOnto/explanationexperience#hasAIModel"]["http://www.w3id.org/iSeeOnto/evaluation#annotatedBy"]

        evals = []
        for temp in assessments:
            metric = temp["http://sensornet.abdn.ac.uk/onts/Qual-O#basedOn"].split("#")[-1].replace("_", " ")         
            value = temp["http://www.w3.org/ns/prov#value"]["value"]
            evals.append([metric, value])

        selected_evals = []
        if selected_metrics:
            selected_evals = [[k,v] for k,v in evals if k.lower() in selected_metrics]
            if len(selected_evals) > 0:
                result_df = pd.DataFrame(np.array(selected_evals), columns=["Assessment Metric", "Value"])
                str_html= result_df.to_html(index=False)+'<br>'
                response={"type":"html", "explanation":str_html}
                return response
            
        result_df = pd.DataFrame(np.array(evals), columns=["Assessment Metric", "Value"])
        str_html= result_df.to_html(index=False)+'<br>'
        response={"type":"html", "explanation":str_html}
        return response

    def get(self,id=None):
        return {
        "_method_description": "Rule based explainer that extracts the performance metrics from case structure representation of the use case.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "selected_metrics" : "(Optional) Array of performance metrics required. Default returns all available.",
            },
        "output_description":{
                "html": "This explanation presents the perfromance metrics of the AI System."
               },
        "meta":{
                "modelAccess":"Any",
                "supportsBWImage":True,
                "needsTrainingData": False


        }
    }
