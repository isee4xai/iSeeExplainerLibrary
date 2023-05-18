import sys
import os
import json
import markdown
import markdown.extensions.fenced_code
from flask import Flask, send_from_directory, make_response
from flask_restful import Api
from flask_cors import CORS
from explainerslist import Explainers
from utils.nlp_explainer_comp import NLPExplainerComparison

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
from resources.explainers.tabular.shapSummary import ShapSummary
from resources.explainers.tabular.shapDependence import ShapDependence
from resources.explainers.tabular.shapInteraction import ShapInteraction
from resources.explainers.tabular.confusionMatrix import ConfusionMatrix
from resources.explainers.tabular.liftCurve import LiftCurve
from resources.explainers.tabular.rocAuc import ROCAUC
from resources.explainers.tabular.prAuc import PRAUC     
from resources.explainers.tabular.precisionGraph import PrecisionGraph
from resources.explainers.tabular.cumulativePrecision import CumulativePrecision
from resources.explainers.tabular.summaryMetrics import SummaryMetrics
from resources.explainers.tabular.regressionPredictedVsActual import RegressionPredictedVsActual
from resources.explainers.tabular.regressionResiduals import RegressionResiduals
from resources.explainers.images.lime import LimeImage
from resources.explainers.images.anchors import AnchorsImage
#from resources.explainers.images.counterfactuals import CounterfactualsImage
from resources.explainers.images.gradcam import GradCam
from resources.explainers.images.integratedGradients import IntegratedGradientsImage
from resources.explainers.images.nn import NearestNeighboursImage
from resources.explainers.text.lime import LimeText
from resources.explainers.text.nlp_clf_explainer import NLPClassifierExpl
from resources.explainers.timeseries.cbrFox import CBRFox
from resources.explainers.timeseries.iGenCBR import IGenCBR
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
api.add_resource(NLPExplainerComparison,'/NLPExplainerComparison')
api.add_resource(DicePublic, '/Tabular/DicePublic','/Tabular/DicePublic/<id>',resource_class_kwargs=path_dict)
api.add_resource(DicePrivate, '/Tabular/DicePrivate','/Tabular/DicePrivate/<id>',resource_class_kwargs=path_dict)
api.add_resource(Lime, '/Tabular/LIME','/Tabular/LIME/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapKernelLocal, '/Tabular/KernelSHAPLocal','/Tabular/KernelSHAPLocal/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapKernelGlobal, '/Tabular/KernelSHAPGlobal','/Tabular/KernelSHAPGlobal/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapTreeLocal, '/Tabular/TreeSHAPLocal','/Tabular/TreeSHAPLocal/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapTreeGlobal, '/Tabular/TreeSHAPGlobal','/Tabular/TreeSHAPGlobal/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapDeepLocal, '/Tabular/DeepSHAPLocal','/Tabular/DeepSHAPLocal/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapDeepGlobal, '/Tabular/DeepSHAPGlobal','/Tabular/DeepSHAPGlobal/<id>',resource_class_kwargs=path_dict)
api.add_resource(Anchors, '/Tabular/Anchors','/Tabular/Anchors/<id>',resource_class_kwargs=path_dict)
api.add_resource(Ale, '/Tabular/ALE','/Tabular/ALE/<id>',resource_class_kwargs=path_dict)
api.add_resource(Importance, '/Tabular/Importance','/Tabular/Importance/<id>',resource_class_kwargs=path_dict)
api.add_resource(DisCERN, '/Tabular/DisCERN','/Tabular/DisCERN/<id>',resource_class_kwargs=path_dict)
api.add_resource(IREX, '/Tabular/IREX','/Tabular/IREX/<id>',resource_class_kwargs=path_dict)
api.add_resource(Nice, '/Tabular/NICE','/Tabular/NICE/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapSummary, '/Tabular/SHAPSummary','/Tabular/SHAPSummary/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapDependence, '/Tabular/SHAPDependence','/Tabular/SHAPDependence/<id>',resource_class_kwargs=path_dict)
api.add_resource(ShapInteraction, '/Tabular/SHAPInteraction','/Tabular/SHAPInteraction/<id>',resource_class_kwargs=path_dict)
api.add_resource(ConfusionMatrix, '/Tabular/ConfusionMatrix','/Tabular/ConfusionMatrix/<id>',resource_class_kwargs=path_dict)
api.add_resource(LiftCurve, '/Tabular/LiftCurve','/Tabular/LiftCurve/<id>',resource_class_kwargs=path_dict)
api.add_resource(ROCAUC, '/Tabular/ROC-AUC','/Tabular/ROC-AUC/<id>',resource_class_kwargs=path_dict)
api.add_resource(PRAUC, '/Tabular/PR-AUC','/Tabular/PR-AUC/<id>',resource_class_kwargs=path_dict)
api.add_resource(PrecisionGraph, '/Tabular/PrecisionGraph','/Tabular/PrecisionGraph/<id>',resource_class_kwargs=path_dict)
api.add_resource(CumulativePrecision, '/Tabular/CumulativePrecision','/Tabular/CumulativePrecision/<id>',resource_class_kwargs=path_dict)
api.add_resource(SummaryMetrics, '/Tabular/SummaryMetrics','/Tabular/SummaryMetrics/<id>',resource_class_kwargs=path_dict)
api.add_resource(RegressionPredictedVsActual, '/Tabular/RegressionPredictedVsActual','/Tabular/RegressionPredictedVsActual/<id>',resource_class_kwargs=path_dict)
api.add_resource(RegressionResiduals, '/Tabular/RegressionResiduals','/Tabular/RegressionResiduals/<id>',resource_class_kwargs=path_dict)
api.add_resource(LimeImage, '/Images/LIME','/Images/LIME/<id>',resource_class_kwargs=path_dict)
api.add_resource(AnchorsImage, '/Images/Anchors','/Images/Anchors/<id>',resource_class_kwargs=path_dict)
#api.add_resource(CounterfactualsImage, '/Images/Counterfactuals','/Images/Counterfactuals/<id>',resource_class_kwargs=path_dict)
api.add_resource(GradCam, '/Images/GradCam','/Images/GradCam/<id>',resource_class_kwargs=path_dict)
api.add_resource(IntegratedGradientsImage, '/Images/IntegratedGradients','/Images/IntegratedGradients/<id>',resource_class_kwargs=path_dict)
api.add_resource(NearestNeighboursImage, '/Images/NearestNeighbours','/Images/NearestNeighbours/<id>',resource_class_kwargs=path_dict)
api.add_resource(LimeText, '/Text/LIME','/Text/LIME/<id>',resource_class_kwargs=path_dict)
api.add_resource(NLPClassifierExpl, "/Text/NLPClassifier",'/Text/NLPClassifier/<id>', resource_class_kwargs=path_dict)
api.add_resource(CBRFox, "/Timeseries/CBRFox",'/Timeseries/CBRFox/<id>', resource_class_kwargs=path_dict)
api.add_resource(IGenCBR, "/Timeseries/iGenCBR",'/Timeseries/iGenCBR/<id>', resource_class_kwargs=path_dict)
api.add_resource(AIModelPerformance, "/Misc/AIModelPerformance",'/Misc/AIModelPerformance/<id>', resource_class_kwargs=path_dict)


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
