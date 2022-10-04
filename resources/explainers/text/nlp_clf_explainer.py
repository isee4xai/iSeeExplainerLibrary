# -*- coding: utf-8 -*-


from flask_restful import Resource,reqparse
import json
from getmodelfiles import get_model_files
from NLPClassifierExplainer.NLPClassifierModel import NLPClassifier
from string import Template



def _generate_html(explanation):
    # ------------------ header section ---------------------

    html = '''
    <html>
    <head>
    </head>
    <body>

    <style>
    .class-name {font-family: monospace}
    .xp-table { border: 2px solid black; border-collapse: collapse}
    .xp-table tr {  border: 1px solid black}
    .xp-table ul {padding: 0px}
    .xp-table td {text-align: left; padding: 20px}
    .xp-table td:first-child {background-color: #f7f7f7; font-weight:bolder; width:25%} 
    .xp-list { display: flex; padding: 10px 10px 10px 0px; gap:15px}
    .progressbar-text { line-height: 30px;}
    .progressbar-bg { background-color: rgb(241, 241, 255); width: 500px; height: 30px}
    .progressbar { background-color: rgb(14, 43, 1); height: 100%; color: white; text-indent: 10px}
    .xp-green {background-color: #e1f5c6;}
    .xp-red {background-color: #f5d9d6;}
    </style>

    <h1>Explanation details</h1>
    <table class="xp-table">
        <tr>
            <td>Confidence scores: similarity per class</td>
            <td>
                <ul>
    '''

    # ---------------  Similarity per class section ---------------
    template = '''
    <li class="progressbar-text xp-list">class <div class="class-name"> ${CLASSNAME}</div> <div class="progressbar-bg">
        <div class="progressbar" style="width: ${PCT}%">${PCT}%</div>
    </div>
    </li>   
    '''

    for cl in explanation['similarity_per_class'].keys():
        s = Template (template)
        html += s.substitute (CLASSNAME=cl, PCT="{:.0f}".format (explanation['similarity_per_class'][cl]))

    html += '''
                </ul>
            </td>
        </tr>
    '''


    # ---------------  keywords  section ---------------
    html +='''
    <tr>
            <td>Top keywords used in the query with TF-IDF score</td>
            <td>
                <div class="xp-list">
    '''

    template = '<span>${KW} (${SCORE})</span>'
    for kw in explanation['keywords'].keys() :
        s = Template (template)
        html += s.substitute (KW=kw, SCORE="{:.3f}".format (explanation['keywords'][kw]))

    html +='''
                </div>
            </td>
        </tr>

    '''

    # ---------------  keywords per class  section ---------------
    html +='''
    <tr>
            <td>Top keywords used in similar texts per class</td>
            <td>
                <ul>
    '''

    for cl in explanation['keywords_per_class'].keys() :
        s1 = Template ('<li class="xp-list">class <span class="class-name">${CL}</span>:')
        html += s1.substitute (CL=cl)

        for kw in explanation['keywords_per_class'][cl].keys():
            s2 = Template ('<span>${KW} (${SCORE}) </span>')
            html += s2.substitute (KW=kw, SCORE="{:.3f}".format (explanation['keywords_per_class'][cl][kw]))

        html +='</li>'



    html +='''
                </div>
            </td>
        </tr>

    '''


    # ---------------  overlapping terms  ---------------
    html +='''
    <tr>
            <td>Overlapping words with similar texts for each class</td>
            <td>
                <ul>
    '''
    for cl in explanation['overlap'].keys() :
        s1 = Template ('<li class="xp-list">class <span class="class-name">${CL}</span>:')
        html += s1.substitute (CL=cl)

        for kw,overlap in explanation['overlap'][cl]:
            s2 = Template ('<span class="${CLASS}">${KW}</span>')
            css_style = "xp-green" if overlap else "xp-red"
            html += s2.substitute (KW=kw, CLASS= css_style)

        html +='</li>'


    html +='''
                </ul>
            </td>
        </tr>
    '''

    # ----------- end  ----------------
    html += '''
    </table>
    </body>
    </html>
    '''

    return html



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
        if(params is not None):
            params_json = json.loads(params)


        #Getting model info, data, and file from local repository
        model_file, _, _ = get_model_files(_id,self.model_folder)

        NLP_model = NLPClassifier()
        NLP_model.load_model (model_file)
        explanation = NLP_model.explain(instance)

        if params_json.get ('format', None) == 'html':
            html = _generate_html(explanation)
            return html

        return explanation

    def get(self):
        return {
        "_method_description": "An explainer for NLP classification models. ",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "A string with the text to be explained."
        }