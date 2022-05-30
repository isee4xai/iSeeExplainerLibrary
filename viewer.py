from flask_restful import Resource
from werkzeug.utils import secure_filename

class ViewExplanation(Resource):
    def get(self, filename):
        return {  
            "filename": secure_filename(filename),
        }