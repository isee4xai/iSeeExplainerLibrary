# ExplainerLibraries

This repository comprises XAI methods (Explainers) used in the iSee Project.

## Want to include your Explainer?

If you want to contribute by including an explainer in the repository, please have a look at the notebooks in the **Explainer Template** folders. Once you have adapted the explainer following the notebook, please reach out to us via Github issues so we can integrate it to the platform. You can also open the notebook in colab by clicking on the button below:

<a target="_blank" href="https://colab.research.google.com/github/isee4xai/iSeeExplainerLibrary/blob/dev/Explainer%20Template/iSee_Explainer_Template.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Using the API with Postman

This quick guide illustrates how to make requests to the ExplanationLibraries server or to a local server, and executing the explanation methods in the API using Postman. 

<!-- #### Launching with Docker

1) Clone the repository.

2) From the root folder, build the Docker image.

```
docker build --tag explainerlibraries .
```

3) Run the container.

```
docker run -p 5000:5000 --name explainerlibrariescont explainerlibraries
```
-->

#### Using the ExplainerLibraries server via Run in Postman

You can use our deployed server to test some example request in our public Postman collection. Simply click the button below and try some of the different requests as explained in the following sections.

[![Run in Postman](https://run.pstmn.io/button.svg)](https://app.getpostman.com/run-collection/18308093-d7be39b5-d207-424e-a272-653f7379348b?action=collection%2Ffork&collection-url=entityId%3D18308093-d7be39b5-d207-424e-a272-653f7379348b%26entityType%3Dcollection%26workspaceId%3D47b431d6-44a4-4fba-a47e-760dfd1447ad)

#### Launching with Python in your Local System

If you are planning to add your own explainer methods to the library, you may want to test them by launching the server on your local machine. To launch the server locally, refer to the following steps:

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
    
3) To execute the methods and get actual explanations, we have to make a POST request. To do so, change the request type to POST and go to *Body > form-data*. Here is where we specify the required parameters, such as the *id*, *url*, and the *params* object. These parameters are explained in greater detail below in the section *About the parameters*. In this example, I am using the psychology model available in the Models folder. The only parameter passed in this case was the *id*.
![Screenshot (157)](https://user-images.githubusercontent.com/71895708/174874200-d99fa678-42ae-4355-9448-53fda3901a87.png)

#### Visualizing Explanations

The responses to the HTTP requests are given in JSON format. However, most of the methods return responses that also contain the URLs to plots or graphs of the explanations in HTML or PNG format. Before accessing the explanations, it is necessary to change the default JSON mime-type.

1) To visualize these explanations, click on the URL in the response. It will open a new request tab with the specified URL.
2) Go to *Headers* and disable the *Accept* attribute.
3) Add a new header with the same name, *Accept*, as a key and specify the value according to the type of file you are trying to access. For .png files, specify *image/png*. For .html files, specify *text/html*. Finally, send the request.
    
![Screenshot (158)](https://user-images.githubusercontent.com/71895708/174875691-fe9509e0-8281-4890-953b-7d88c5e87a69.png)

## About the Parameters

The required parameters may be different depending on the explainer, so it is recommended to see the documentation provided by the get method of the explainer being used.

- **id**: the *id* is a 10-character long string composed of letters and/or numbers. It is used to access the server space dedicated to the model to be explained. This space is a folder with the same name as the id located in the *Models* folder. This folder is created by the "Model AI Library" when a user uploads a model file (or an external URL), the training data (if required), and specific information about the model. Note that **if you want to use your own model**, you a folder with the followinf files to the "Models" folder:
	- _Model File_: The trained prediction model given as a compressed file. The extension must match the backend being used i.e. a .pkl file for Scikit-learn (use Joblib library), .pt for PyTorch, or .h5 for TensorFlow models. The name of the files must be the same as the id. For models with different backends, it is possible to upload a .pkl, but it is necessary that the prediction function of the model is called 'predict'. 
	- _Data File_: Pandas DataFrame containing the training data given as a .pkl file (use Joblib library). The name of this file must be the id concatenates with the string "_data", i.e: PSYCHOLOGY_data.pkl. The target class must be the last column of the DataFrame. Currently, it is only needed for tabular data models.
	- _Model Info_: JSON file containing the characteristics of the model, also referred to as model attributes. Some attributes ar mandatory, such as the alias of the model, which is the common name that will be assigned to it. Also the backend and the task performed by the model are required in most cases. Other attributes are optional, such as the names of the features for tabular data, the categorical features, the labels of the output classes, etc. Even though some of these attributes may be optional, they may considerably improve the quality of the explanation, mostly from a visualization point of view. Note that model attribues are *static*, they don't vary from execution to execution. Please refer to the [model_info_attributes.txt](model_info_attributes.txt) file to see the currently defined attributes among all the explainers available.

	**Note:** Regardless of the uploaded files, **all the methods require an id to be provided. If you want to test a method with your own model, upload a folder containing the previously described files to the Models folder, assigning an id of your choice**. See the example below for a model with id "PSYCHOLOGY".
	
	
	
	<p align="center">
  <img src="https://user-images.githubusercontent.com/71895708/191551057-2a043e68-51b0-4304-bace-4db6b9ba12d4.png"/>
</p>


- **url**: External URL of the prediction function passed as a string. This parameter provides an alternative when the model owners do not want to upload the model file and the explanation method is able to work with a prediction function instead of a model object. **The URL is ignored if a model file was uploaded to the server**. This related server must be able to handle a POST request receiving a multi-dimensional array of N data points as inputs (instances represented as arrays). It must return an array of N outputs (predictions for each instance). Refer to the _External URLs Examples folder_ if you want to quickly create a service using Flask to provide this method. Please see the example in the section below.

- **instance**: This is a mandatory attribute for local methods, as it is the instance that will be explained. It is an array containing the feature values (which must be in the same order that the model expects). For images, it is a matrix representing the pixels. It is also possible for image explainers to pass a file instead of the matrix using the "image" argument.

- **params**: dictionary with the specific execution parameters passed to the explanation method. These parameters are optional and depend on the method being used. The value assigned to this parameters may signficantly change the outcome of an explanation. For example, the "target_class" of a counterfactual is an execution parameter. Refer to the documentation of each method to know the configuration parameters that can be provided.

## Getting Explanations Using External URLs Models

In some cases, uploading a model file to the server is not desired by the user or simply not possible. Some explanation methods provide an alternative, as they only need access to the prediction function of the model. The prediction function can be easily wrapped as an HTTP POST method so that the explainers can access the prediction function by making requests to a server administered by the user. However, the implementation of the POST method must follow the expected format:

- **The POST method must receive a parameter named "inputs" and return an array with the predictions**. The format of the "inputs" parameter, as well as the output, must be as follows:

  - For Tabular and Text models:
  	- For Regression Models:
		- inputs: array of shape *(n, f)* where *n* is the number of instances and *f* is the number of features.
		- output: array of shape *(n,)* where *n* is the number of instances. Contains the predicted value for each instance.
	- For Classification Models:
		- inputs: array of shape *(n, f)* where *n* is the number of instances and *f* is the number of features.
		- output: array of shape *(n, c)* where *n* is the number of instances and *c* is the number of classes. Contains the predicted probabilities of each class for each instance.

  - For Image models: 
  	- inputs: Array of shape *(n, h, w)* for black and white images, and shape *(n, h, w, 3)* for RGB images, where *n* is the number of images, *h* the pixel height, and *w* the pixel width.
	- output: array of shape *(n, c)* where *n* is the number of instances and *c* is the number of classes. Contains the predicted probabilities of each class for each image.
	
Notice that if you are using a model from Tensorflow or Scikit-learn, the *predict* or *predict_proba* function of your model already matches this format. If you have models from different architectures, some additional wrapping code may be necessary to comply with this format.

For illustration purposes, we will implement the POST method with Flask using the psychology model. Example implementations of external URL prediction functions are available in the *External_URLs* folder.

**1)** If you are testing locally, launch the explainer libraries server as described before. 

**2)** For the server logic, load the previously trained model first. Then define the POST method and add the inputs parameter to the parser. Load the contents of the inputs parameter and pass them to the prediction function of your model. We use the predict_proba function, as the psychology model is a scikit-learn classifier. Finally, specify the path for the method by adding it to the API. **Note**: if you are testing locally, make sure to assign a different port from the explainer libraries server.

```python
import sys
from flask import Flask
from flask_restful import Api,Resource,reqparse
import numpy as np
import json
import joblib

cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
app = Flask(__name__)
api = Api(app)

#Load the model
model = joblib.load("PSYCHOLOGY.pkl")

class Predict(Resource):
   
    def post(self):
        #Add the 'inputs' argument
        parser = reqparse.RequestParser()
        parser.add_argument("inputs", required=True)
        args = parser.parse_args()
        
        #Get the inputs and pass them to the prediction function
        inputs = np.array(json.loads(args.get("inputs")))
        return model.predict_proba(inputs).tolist()

# Add the resource to the API
api.add_resource(Predict, '/Predict')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
```

**3)** Run the server and test the POST method by passing the *url* parameter to the explanation method. Remember that the url is ignored if a model file was uploaded to the server, so make sure no model file is present in the corresponding folder.

![Screenshot (166)](https://user-images.githubusercontent.com/71895708/182601845-7051449b-1cc0-4fa3-8ba9-083831b58f23.png)


## How to Collaborate to ExplainerLibraries

**1)** Fork the repo and clone it to our local machine. 

**2)** Create our own branch.

**3)** Add the explainer file and make the necessary modifications.

**4)** Launch the application locally and test the new explainer.

**5)** Push the changes and create a pull request for review.


## Adding New Explainers to the Catalogue

**1)**	To add a new explainer, it is necessary to create a new Resource. First, go to the _resources/explainers_ folder and select the folder corresponding to the data type of the explainer you want to add (If your explainer works with a different data type, please add the corresponding folder to the resources folder). For illustration purposes, we will walk through the steps of adding a "new" explainer (LIME tabular).

**2)**	Inside the appropriate folder, ***create a new .py file*** with the name of your explainer. In our case, we create the lime.py file inside _resources/explainers/tabular/_ .

**3)**	Create a class for the explainer. This class needs to have ***two different methods: post and get***. You may also need to add an **\_\_init\_\_** method to access the paths of the models and uploads folders. In our example:

```python
from flask_restful import Resource

class Lime(Resource):

    	def __init__(self,model_folder,upload_folder):
		self.model_folder = model_folder
		self.upload_folder = upload_folder  
	
	def post(self):
		return {}
		
	def get(self):
		return {}
```


**5)**	In the **post method**, define the mandatory arguments that must be passed for the explainer to get an explanation. The method must receive at least an id to access the folder related to the model. After parsing the arguments, use the function _get_model_files_, passing the id to fetch the model, data, and info files. It is possible that some of these files do not exist, so make the appropriate checks before using them. Generally, the steps involve loading the Dataframe with the training data if it exists, then getting the necessary attributes from the info file, then getting the prediction function if possible, and finally getting the configuration parameters from the _params_ object.

```python	
class Lime(Resource):

def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id',required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument('url')
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## loading data
        if data_file!=None:
            dataframe = joblib.load(data_file) ##error handling?
        else:
            raise Exception("The training data file was not provided.")

        ##getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        kwargsData = dict(mode="classification", feature_names=None, categorical_features=None,categorical_names=None, class_names=None)
        if "model_task" in model_info:
            kwargsData["mode"] = model_info["model_task"]
        if "feature_names" in model_info:
            kwargsData["feature_names"] = model_info["feature_names"]
        if "categorical_features" in model_info:
            kwargsData["categorical_features"] = model_info["categorical_features"]
        if "categorical_names" in model_info:
            kwargsData["categorical_names"] = {int(k):v for k,v in model_info["categorical_names"].items()}
        if "output_names" in model_info:
            kwargsData["class_names"] = model_info["output_names"]

        ## getting predict function
        predic_func=None
        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend=="sklearn":
                mlp = joblib.load(model_file)
                predic_func=mlp.predict_proba
            elif backend=="PYT":
                mlp = torch.load(model_file)
                predic_func=mlp.predict
            else:
                mlp = joblib.load(model_file)
                predic_func=mlp.predict
        elif url!=None:
            def predict(X):
                return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
            predic_func=predict
        else:
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")

  
        
        #getting params from request
        kwargsData2 = dict(labels=(1,), top_labels=None, num_features=None)
        if "output_classes" in params_json: #labels
            kwargsData2["labels"] = json.loads(params_json["output_classes"]) if isinstance(params_json["output_classes"],str) else params_json["output_classes"]  
        if "top_classes" in params_json:
            kwargsData2["top_labels"] = int(params_json["top_classes"])   #top labels
        if "num_features" in params_json:
            kwargsData2["num_features"] = int(params_json["num_features"])

	...
```
**6)** Add the actual code for the generation of the explanation to the post method. This depends entirely on the explanation method being used. Once the explanation has been created, convert it to a JSON format if necessary. If the explanation is returned as an html or png file, use the save_file_info function to get the upload folder path, the name that will be given to the file, and the url (getcall) that will be used to access the file. Save the file using this data and append the URL to the returned JSON. **Note:** the URL to access the file returned by save_file_info does not include the extension of the file, so it is necessary to append it at the end as it is shown in the example.

```python	
class Lime(Resource):

def post(self):

	...
	        
	explainer = lime.lime_tabular.LimeTabularExplainer(dataframe.drop(dataframe.columns[len(dataframe.columns)-1], axis=1, inplace=False).to_numpy(),
                                                            **{k: v for k, v in kwargsData.items() if v is not None})
        explanation = explainer.explain_instance(np.array(instance, dtype='f'), predic_func, **{k: v for k, v in kwargsData2.items() if v is not None}) 
        
        ## Formatting json explanation
        ret = explanation.as_map()
        ret = {str(k):[(int(i),float(j)) for (i,j) in v] for k,v in ret.items()}
        if kwargsData["class_names"]!=None:
            ret = {kwargsData["class_names"][int(k)]:v for k,v in ret.items()}
        if kwargsData["feature_names"]!=None:
            ret = {k:[(kwargsData["feature_names"][i],j) for (i,j) in v] for k,v in ret.items()}
        ret=json.loads(json.dumps(ret))

        ## saving to Uploads
        upload_folder, filename, getcall = save_file_info(request.path)
        hti = Html2Image()
        hti.output_path= upload_folder
        hti.screenshot(html_str=explanation.as_html(), save_as=filename+".png")   
        explanation.save_to_file(upload_folder+filename+".html")
        
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":ret}
        return response
```
**7)** For the get method, return a dictionary that serves as documentation for the explainer that is being implemented. In our implementations, we include a brief description of the explainer method and the parameters to the request, as well as the configuration parameters that should be passed in the _params_ dictionary. If necessary, we also include an example of the _params_ object. For example, for the Tabular/LIME implementation:

```python
    def get(self):
                return {
        "_method_description": "LIME perturbs the input data samples in order to train a simple model that approximates the prediction for the given instance and similar ones. "
                           "The explanation contains the weight of each attribute to the prediction value. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url'(optional),  and the 'params' dictionary (optiohnal) with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array representing a row with the feature values of an instance not including the target class.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "output_classes" : "(Optional) Array of ints representing the classes to be explained.",
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probability to be explained. Overrides 'output_classes' if provided.",
                "num_features": "(Optional) Int representing the maximum number of features to be included in the explanation."
                }

```
**8)** Lastly, add the class as a resource and specify its route in the _app.py_ and in the _explainerslist.py_ files. __Also update the model_info_attributes.txt__ file if you are using a new model attribute that was not included before. In our example:

```python
from resources.explainers.tabular.lime import Lime
api.add_resource(Lime, '/Tabular/LIME')
```



## Found an Issue?

Please open a GitHub issue with as much detail as possible and we will try to fix it as soon as we can. Thank you!
