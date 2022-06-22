import sys
from flask import Flask
from flask_restful import Api,Resource,reqparse
import tensorflow as tf
import numpy as np
import h5py
import json

cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
app = Flask(__name__)
api = Api(app)


class Predict(Resource):
   
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("inputs", required=True)
        args = parser.parse_args()
        inputs = np.array(json.loads(args.get("inputs")))
        model = tf.keras.models.load_model("mnist_cnn.h5")
        return model.predict(inputs).tolist()


api.add_resource(Predict, '/Predict')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)
