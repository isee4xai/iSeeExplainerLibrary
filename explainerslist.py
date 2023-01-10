from flask_restful import Resource

class Explainers(Resource):
    def get(self):
        return [ '/Images/Anchors',
                 '/Images/Counterfactuals',
                 '/Images/GradCamTorch',
                 '/Images/IntegratedGradients',
                 '/Images/LIME',
                 '/Tabular/ALE',
                 '/Tabular/Anchors',
                 '/Tabular/DeepSHAPGlobal',
                 '/Tabular/DeepSHAPLocal',
                 '/Tabular/DicePrivate',
                 '/Tabular/DicePublic',
                 '/Tabular/DisCERN',
                 '/Tabular/IREX',
                 '/Tabular/Importance',
                 '/Tabular/KernelSHAPGlobal',
                 '/Tabular/KernelSHAPLocal',
                 '/Tabular/LIME',
                 '/Tabular/NICE',
                 '/Tabular/TreeSHAPGlobal',
                 '/Tabular/TreeSHAPLocal',
                 '/Text/LIME',
                 '/Text/NLPClassifier',
                 '/Timeseries/CBRFox']