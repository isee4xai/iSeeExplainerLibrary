from flask_restful import Resource
from flask import request
import requests
import json

class Meta(Resource):

    def get(self, datatype, explainer):
        
        url=request.url_root+datatype+"/"+explainer
        response=requests.get(url)
        if response.ok:
           jsondocs=json.loads(response.text)
        else:
            raise Exception("Could not load available explainers.")
        ret={}
        if "meta" in jsondocs:
            ret=jsondocs["meta"]
        return ret
