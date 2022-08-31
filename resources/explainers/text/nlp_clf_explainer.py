# -*- coding: utf-8 -*-


from flask_restful import Resource,reqparse
import json
from getmodelfiles import get_model_files
from NLPClassifierExplainer.NLPClassifierModel import NLPClassifier

class NLPClassifierExpl(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        instance = args.get("instance")
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)


        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        NLP_model = NLPClassifier()
        NLP_model.load_model (model_file)
        
    
        explanation = NLP_model.explain(instance)
        
        return explanation

    def get(self):
        return {
        "_method_description": "An explainer for NLP classification models. ",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "A string with the text to be explained."
    
        }