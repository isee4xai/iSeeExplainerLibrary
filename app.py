import sys
from flask import Flask
from flask_restful import Api

from resources.tabular.dicePublic import DicePublic
from resources.tabular.dicePrivate import DicePrivate
from resources.tabular.lime import Lime
from resources.tabular.shap import Shap
from resources.tabular.anchors import Anchors
from resources.tabular.ale import Ale
from resources.tabular.importance import Importance

cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
app = Flask(__name__)
api = Api(app)

api.add_resource(DicePublic, '/Tabular/DicePublic')
api.add_resource(DicePrivate, '/Tabular/DicePrivate')
api.add_resource(Lime, '/Tabular/LIME')
api.add_resource(Shap, '/Tabular/SHAP')
api.add_resource(Anchors, '/Tabular/Anchors')
api.add_resource(Ale, '/Tabular/ALE')
api.add_resource(Importance, '/Tabular/Importance')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
