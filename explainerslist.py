from flask_restful import Resource

class Explainers(Resource):
    def get(self):
        return [ '/Images/Anchors',
                 '/Images/Counterfactuals',
                 '/Images/GradCam',
                 '/Images/IntegratedGradients',
                 '/Images/LIME',
                 '/Images/NearestNeighbours',
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
                 '/Timeseries/CBRFox',
                 '/Misc/AIModePerformance']
