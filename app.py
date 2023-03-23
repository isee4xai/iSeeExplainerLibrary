import sys
import os
import json
import markdown
import markdown.extensions.fenced_code
from flask import Flask, send_from_directory, make_response, request
from flask_restful import Api
from flask_cors import CORS
from viewer import ViewExplanation
from explainerslist import Explainers

from resources.explainers.tabular.dicePublic import DicePublic 
from resources.explainers.tabular.dicePrivate import DicePrivate
from resources.explainers.tabular.lime import Lime
from resources.explainers.tabular.shapKernelLocal import ShapKernelLocal
from resources.explainers.tabular.shapKernelGlobal import ShapKernelGlobal
from resources.explainers.tabular.shapTreeLocal import ShapTreeLocal
from resources.explainers.tabular.shapTreeGlobal import ShapTreeGlobal
from resources.explainers.tabular.shapDeepLocal import ShapDeepLocal
from resources.explainers.tabular.shapDeepGlobal import ShapDeepGlobal
from resources.explainers.tabular.anchors import Anchors
from resources.explainers.tabular.ale import Ale
from resources.explainers.tabular.importance import Importance
from resources.explainers.tabular.discern import DisCERN
from resources.explainers.tabular.irex import IREX
from resources.explainers.tabular.nice import Nice
from resources.explainers.images.lime import LimeImage
from resources.explainers.images.anchors import AnchorsImage
from resources.explainers.images.counterfactuals import CounterfactualsImage
from resources.explainers.images.gradcam import GradCam
from resources.explainers.images.integratedGradients import IntegratedGradientsImage
from resources.explainers.images.nn import NearestNeighboursImage
from resources.explainers.text.lime import LimeText
#from resources.explainers.text.nlp_clf_explainer import NLPClassifierExpl
from resources.explainers.timeseries.cbrFox import CBRFox
from resources.explainers.misc.performance import AIModelPerformance


MODEL_FOLDER="Models"
UPLOAD_FOLDER="Uploads"
if len(sys.argv) > 3 :
    raise Exception("Too many arguments passed to the program")
else:
    if len(sys.argv) >= 2 :
        if os.path.exists(sys.argv[1]):
            if os.path.isdir(sys.argv[1]):
                print("Using existing directory '" +sys.argv[1]+ "'")
            else:
                raise Exception("A non-directory file named '" + sys.argv[1]+ "' already exists. Please use another name.")
        else:
            os.mkdir(sys.argv[1])
            print("The '" +sys.argv[1]+ "' directory was created.")
        MODEL_FOLDER=sys.argv[1]
    else:
        if os.path.exists(MODEL_FOLDER):
            if os.path.isdir(MODEL_FOLDER):
                print("Using existing default directory '" +MODEL_FOLDER+ "'")
            else:
                raise Exception("A non-directory file named '" + MODEL_FOLDER+ "' already exists. Please use another name.")
        else:
            os.mkdir(MODEL_FOLDER)
            print("The '" +MODEL_FOLDER+ "' default directory was created.")

    if len(sys.argv) == 3:
        if os.path.exists(sys.argv[2]):
            if os.path.isdir(sys.argv[2]):
                print("Using existing directory '" +sys.argv[2]+ "'")
            else:
                raise Exception("A non-directory file named '" + sys.argv[2]+ "' already exists. Please use another name.")
        else:
            os.mkdir(sys.argv[2])
            print("The '" +sys.argv[2]+ "' directory was created.")
        UPLOAD_FOLDER=sys.argv[2]
    else:
        if os.path.exists(UPLOAD_FOLDER):
            if os.path.isdir(UPLOAD_FOLDER):
                print("Using existing default directory '" +UPLOAD_FOLDER+ "'")
            else:
                raise Exception("A non-directory file named '" + UPLOAD_FOLDER+ "' already exists. Please use another name.")
        else:
            os.mkdir(UPLOAD_FOLDER)
            print("The '" +UPLOAD_FOLDER+ "' default directory was created.")
path_dict={"model_folder":MODEL_FOLDER,"upload_folder":UPLOAD_FOLDER}
    
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
app = Flask(__name__)




app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
api = Api(app)
api.add_resource(Explainers,'/Explainers')
#api.add_resource(ViewExplanation, '/ViewExplanation/<string:filename>')
api.add_resource(DicePublic, '/Tabular/DicePublic',resource_class_kwargs=path_dict)
api.add_resource(DicePrivate, '/Tabular/DicePrivate',resource_class_kwargs=path_dict)
api.add_resource(Lime, '/Tabular/LIME',resource_class_kwargs=path_dict)
api.add_resource(ShapKernelLocal, '/Tabular/KernelSHAPLocal',resource_class_kwargs=path_dict)
api.add_resource(ShapKernelGlobal, '/Tabular/KernelSHAPGlobal',resource_class_kwargs=path_dict)
api.add_resource(ShapTreeLocal, '/Tabular/TreeSHAPLocal',resource_class_kwargs=path_dict)
api.add_resource(ShapTreeGlobal, '/Tabular/TreeSHAPGlobal',resource_class_kwargs=path_dict)
api.add_resource(ShapDeepLocal, '/Tabular/DeepSHAPLocal',resource_class_kwargs=path_dict)
api.add_resource(ShapDeepGlobal, '/Tabular/DeepSHAPGlobal',resource_class_kwargs=path_dict)
api.add_resource(Anchors, '/Tabular/Anchors',resource_class_kwargs=path_dict)
api.add_resource(Ale, '/Tabular/ALE',resource_class_kwargs=path_dict)
api.add_resource(Importance, '/Tabular/Importance',resource_class_kwargs=path_dict)
api.add_resource(DisCERN, '/Tabular/DisCERN',resource_class_kwargs=path_dict)
api.add_resource(IREX, '/Tabular/IREX',resource_class_kwargs=path_dict)
api.add_resource(Nice, '/Tabular/NICE',resource_class_kwargs=path_dict)
api.add_resource(LimeImage, '/Images/LIME',resource_class_kwargs=path_dict)
api.add_resource(AnchorsImage, '/Images/Anchors',resource_class_kwargs=path_dict)
api.add_resource(CounterfactualsImage, '/Images/Counterfactuals',resource_class_kwargs=path_dict)
api.add_resource(GradCam, '/Images/GradCam',resource_class_kwargs=path_dict)
api.add_resource(IntegratedGradientsImage, '/Images/IntegratedGradients',resource_class_kwargs=path_dict)
api.add_resource(NearestNeighboursImage, '/Images/NearestNeighbours',resource_class_kwargs=path_dict)
api.add_resource(LimeText, '/Text/LIME',resource_class_kwargs=path_dict)
#api.add_resource(NLPClassifierExpl, "/Text/NLPClassifier", resource_class_kwargs=path_dict)
api.add_resource(CBRFox, "/Timeseries/CBRFox", resource_class_kwargs=path_dict)
api.add_resource(AIModelPerformance, "/Misc/AIModelPerformance", resource_class_kwargs=path_dict)


@api.representation('image/png')
def output_file_png(data, code, headers):
    response = send_from_directory(UPLOAD_FOLDER,
    data["filename"],mimetype="image/png")
    return response

@api.representation('text/html')
def output_file_html(data, code, headers):
    if isinstance(data,dict) and "filename" in data:
        response = send_from_directory(UPLOAD_FOLDER,
        data["filename"],mimetype="text/html")
    else:
        response = make_response(json.dumps(data), code)
        response.headers.extend(headers or {})
    return response

@app.route("/")
def index():
    readme_file = open('README.md', 'r')
    md_template_string = markdown.markdown(readme_file.read(), extensions=["fenced_code"])
    return md_template_string

@app.route('/ViewExplanation/<string:filename>',methods=['GET'])
def view_explanation(filename):
    if filename is None:
        return "The filename is missing."
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        return "The file does not exist"
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
