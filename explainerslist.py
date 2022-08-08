from flask_restful import Resource
from werkzeug.utils import secure_filename

class Explainers(Resource):
    def get(self):
        return [
                    '/Tabular/DicePublic',
                    '/Tabular/DicePrivate',
                    '/Tabular/LIME',
                    '/Tabular/KernelSHAPLocal',
                    '/Tabular/KernelSHAPGlobal',
                    '/Tabular/TreeSHAPLocal',
                    '/Tabular/TreeSHAPGlobal',
                    '/Tabular/DeepSHAPLocal',
                    '/Tabular/DeepSHAPGlobal',
                    '/Tabular/Anchors',
                    '/Tabular/ALE',
                    '/Tabular/Importance',
                    '/Images/LIME',
                    '/Images/Anchors',
                    '/Images/Counterfactuals',
                    '/Images/GradCamTorch',
                    '/Text/LIME'
                ]