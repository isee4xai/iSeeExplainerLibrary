# ExplainerLibraries

## Using the API with Postman

This quick guide illustrates how to launch the Flask server and make requests to any of the explanation methods in the API using Postman. 

#### Launching the Server

1) Clone the repository.

2) From the root folder, create a virtual environment for the installation of the required libraries with:

           
```console
python -m venv .
```
                
            
3) Use pip to install the dependencies from the requirements file.

```console
pip install -r requirements.txt
```
            
4) Once all the dependencies have been installed, execute the script to launch the server with:

```console
python app.py
```
    
#### Making Requests

If the server was launched successfully, a similar message to the one in the image should appear, meaning that it is ready to receive requests to the specified address and port.

![ServerLaunched](https://user-images.githubusercontent.com/71895708/170830447-760dce21-69b3-4538-ab37-22f6d058ed1f.PNG)

1) To make requests, open Postman and go to *My Workspace > File > New Tab*.
2) To get information about how to use a specific method, we can make a GET request. In the URL bar, specify the address and port of the server, followed by the name of the method, and send the request. The response is displayed in the bottom part of the console. For example, for Tabular/Importance:
    
![Screenshot (119)](https://user-images.githubusercontent.com/71895708/174871434-d57e4d63-ef2a-4513-8deb-8672e1815cb6.png)
    
3) To execute the methods and get actual explanations, we have to make a POST request. To do so, change the request type to POST and go to *Body > form-data*. Here is where we specify the required parameters, such as the *id*, *url*, and the *params* object. These parameters are explained in greater detail below in the section *About the parameters*. In this example, I am using the psychology model available at the Models folder. The only parameter passed in this case was the *id*.
![Screenshot (157)](https://user-images.githubusercontent.com/71895708/174874200-d99fa678-42ae-4355-9448-53fda3901a87.png)

#### Visualizing Explanations

The responses to the HTTP requests are given in JSON format. However, most of the methods return responses that also contain the URLs to plots or graphs of the explanations in HTML or PNG format. Before accessing the explanations, it is necessary to change the default JSON mime-type.

1) To visualize these explanations, click on the URL in the response. It will open a new request tab with the specified URL.
2) Go to *Headers* and disable the *Accept* attribute.
3) Add a new header with the same name, *Accept*, as key and specify the value according to the type of file you are trying to access. For .png files, specify *image/png*. For .html files, specify *text/html*. Finally, send the request.
    
![Screenshot (158)](https://user-images.githubusercontent.com/71895708/174875691-fe9509e0-8281-4890-953b-7d88c5e87a69.png)

## About the Parameters

The required parameters may be different depending on the explainer, so it is recommended to see the documentation provided by the get method of the explainer being used.

- **id**: the *id* is a 10-character long string composed of letters and/or numbers. It is used to access the server space dedicated to the model to be explained. This space is a folder with the same name of the id located in the *Models* folder. This folder is created by the "Model AI Library" when a user uploads a model file (or an external URL), the training data (if required), and specific information about the model:
	- _Model File_: The trained prediction model given as a compressed file. The extension must match the backend being used i.e. a .pkl file for Scikit-learn (use Joblib library), .pt for PyTorch, or .h5 for TensorFlow models. For models with different backends, it is possible to upload a .pkl, but it is necessary that the prediction function of the model is called 'predict'. 
	- _Data File_: Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The target class must be the last column of the DataFrame. Only needed for tabular data models.
	- _Model Info_: Json file containing the characteristics of the model. Some characteristics must be always specified, such as the backend of the model. Others are optional, such as the names of the features for tabular data, the categorical features, the labels of the output classes, etc. Please refer to the *model_info_attr.txt* file to see the currently defined attributes.

Regardless of the provided files, **all the methods require an id to be provided**.

-**url**: External URL of the prediction function. This parameter provides an alternative when the model owners do not want to upload the model file. The URL is ignored if a model file was uploaded to the server. This related server must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance). Refer to the _External URL Examples_ if you want to quickly create a service using flask to implement this method.


## Adding new explainers to the catalogue

**1)**	To add a new explainer, it is necessary to create a new Resource. First, go to the _resources/explainers_ folder and select the folder corresponding to the data type of the explainer you want to add (If your explainer works with a different data type, please add the corresponding folder to the resources folder). For illustration purposes, we will walk through the steps of adding a "new" explainer (LIME tabular).

**2)**	Inside the appropriate folder, ***create a new .py file*** with the name of your explainer. In our case, we create the lime.py file  inside _resources/explainers/tabular/_ .

**3)**	Create a class for the explainer. This class needs to have ***two different methods: post and get***. In our example:

```python
from flask_restful import Resource

class Lime(Resource):

	def post(self):
		return {}
		
	def get(self):
		return {}
```
**4)**	In the **post method**, define the mandatory arguments that must be passed for the explainer to get an explanation. In most explainers, this includes the files for the model and data (when needed), and an additional argument called params, which is a dictionary containing parameters such as a particular instance for local methods, configuration options, and additional information needed by the explainer. Note that in the example that after parsing the arguments, we use joblib to load the file parameters since the model and data are passed as pickled files.

```python	
class Lime(Resource):

def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("model", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument("data", type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('params')
        args = parser.parse_args()
        
        data = args.get("data")
        model = args.get("model")
        dataframe = joblib.load(data)
        params_json = json.loads(args.get("params"))
        instance=params_json["instance"]
        backend = params_json["backend"]

	...
```
**5)** Add the actual code for the generation of the explanation to the post method. **Note:** the returned values must be a dictionary containing the explanation. This dictionary may also contain a URL to download a plot or image of the explanation when applicable.

**6)** For the get method, return a dictionary that serves as documentation for the explainer that is being implemented. In our implementations, we include a brief description of the explainer method and the parameters to the request, as well as the configuration parameters that should be passed in the _params_ dictionary. If necessary, we also include an example of the _params_ object. For example, for the Tabular/LIME implementation:

```python
    def get(self):
        return {
        "_method_description": "LIME perturbs the input data samples in order to train a simple model that approximates the prediction for the given instance and similar ones. "
                           "The explanation contains the weight of each attribute to the prediction value. This method accepts 3 arguments: " 
                           "the 'id', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "instance": "Array representing a row with the feature values of an instance not including the target class.",
                "output_classes" : "(Optional) Array of ints representing the classes to be explained.",
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probability to be explained. Overrides 'output_classes' if provided.",
                "num_features": "(Optional) Int representing the maximum number of features to be included in the explanation."
                }

        }
```
**7)** Lastly, add the class as a resource and specify its route in the _app.py_ file. In our example:

```python
from resources.explainers.tabular.lime import Lime
api.add_resource(Lime, '/Tabular/LIME')
```

