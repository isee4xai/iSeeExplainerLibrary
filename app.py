import sys
from flask import Flask, send_from_directory,request
from flask_restful import Api
import matplotlib.pyplot

from viewer import ViewExplanation
from resources.explainers.tabular.dicePublic import DicePublic
from resources.explainers.tabular.dicePrivate import DicePrivate
from resources.explainers.tabular.lime import Lime
from resources.explainers.tabular.shap import Shap
from resources.explainers.tabular.anchors import Anchors
from resources.explainers.tabular.ale import Ale
from resources.explainers.tabular.importance import Importance
from resources.explainers.images.lime import LimeImage
from resources.explainers.images.anchors import AnchorsImage
from resources.explainers.images.counterfactuals import CounterfactualsImage
from resources.explainers.text.lime import LimeText


UPLOAD_FOLDER="Uploads/"

cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
app = Flask(__name__)
api = Api(app)



api.add_resource(ViewExplanation, '/ViewExplanation/<string:filename>')
api.add_resource(DicePublic, '/Tabular/DicePublic')
api.add_resource(DicePrivate, '/Tabular/DicePrivate')
api.add_resource(Lime, '/Tabular/LIME')
api.add_resource(Shap, '/Tabular/SHAP')
api.add_resource(Anchors, '/Tabular/Anchors')
api.add_resource(Ale, '/Tabular/ALE')
api.add_resource(Importance, '/Tabular/Importance')
api.add_resource(LimeImage, '/Images/LIME')
api.add_resource(AnchorsImage, '/Images/Anchors')
api.add_resource(CounterfactualsImage, '/Images/Counterfactuals')
api.add_resource(LimeText, '/Text/LIME')


@api.representation('image/png')
def output_file_png(data, code, headers):
    response = send_from_directory(UPLOAD_FOLDER,
    data["filename"],mimetype="image/png",as_attachment=True)
    return response

@api.representation('text/html')
def output_file_html(data, code, headers):
    response = send_from_directory(UPLOAD_FOLDER,
    data["filename"],mimetype="text/html",as_attachment=True)
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
