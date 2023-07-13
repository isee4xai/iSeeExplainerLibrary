from http.client import BAD_REQUEST
from flask_restful import Resource
from flask import request
import tensorflow as tf
import pandas as pd
import numpy as np
import h5py
import json
import plotly.express as px
from tensorflow import keras
from io import BytesIO
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from getmodelfiles import get_model_files
from utils import ontologyConstants
from utils.dataframe_processing import split_sequences, normalize_dataframe
from utils.base64 import PIL_to_base64
import traceback



class IGenCBR(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder 
        
    def post(self):
        try:
            params = request.json
            if params is None:
                return "The json body is missing.",BAD_REQUEST
        
            #Check params
            if("id" not in params):
                return "The model id was not specified in the params.",BAD_REQUEST
            if("type" not in params):
                return "The instance type was not specified in the params.",BAD_REQUEST
            if("instance" not in params):
                return "The instance was not specified in the params.",BAD_REQUEST

            _id =params["id"]
            if("type"  in params):
                inst_type=params["type"]
            instance=params["instance"]
            params_json={}
            if "params" in params:
                params_json=params["params"]
        
        
            #Getting model info, data, and file from local repository
            model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

            #loading data
            if data_file!=None:
                dataframe = pd.read_csv(data_file,header=0)
            else:
                return "The training data file was not provided.",BAD_REQUEST

            ##getting params from info
            model_info=json.load(model_info_file)
            backend = model_info["backend"] 
            target_names=model_info["attributes"]["target_names"]
            features=model_info["attributes"]["features"]
            feature_names=list(dataframe.columns)
            k=model_info["attributes"]["window_size"]

            time_feature=None
            for feature, info_feature in features.items():
                if(info_feature["data_type"]=="time"):
                    time_feature=feature
                    break 
            

            if time_feature is not None:
                dataframe.drop([time_feature], axis=1,inplace=True)
                feature_names.remove(time_feature)

            input_features=[]
            for f in feature_names:
                if(f not in target_names):
                    input_features.append(f)
        
            for target in target_names:
                feature_names.remove(target)
                    
            #split into input and target
            input=split_sequences(dataframe.drop(target_names,axis=1),k)
            data_values = dataframe.drop(target_names,axis=1).values


            #loading model
            if model_file!=None:
                if backend in ontologyConstants.TENSORFLOW_URIS:
                    model=h5py.File(model_file, 'w')
                    model = tf.keras.models.load_model(model,compile=False)
                    model.compile()
                else:
                    return "This method currently supports Tensoflow models only.",BAD_REQUEST
            else:
                return "A TensorFlow model file must be provided.",BAD_REQUEST

            #getting params from request
            NUM_NEIGHBOURS = 3
            col = 0
            if "num_neighbours" in params_json: 
                NUM_NEIGHBOURS = int(params_json["num_neighbours"])
            if "feature" in params_json:
                if(params_json["feature"] in feature_names):
                    col = feature_names.index(params_json["feature"])  

            #denormalizing instance
            df_instance=pd.DataFrame(instance)
            df_instance=df_instance[feature_names]

            norm_instance=normalize_dataframe(df_instance, model_info).to_numpy()

            #helpers
            def compute_gradients(input_xs, target_class_idx):
                with tf.GradientTape() as tape:
                    tape.watch(input_xs)
                    logits = model(input_xs)
            #         probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx] # change 2 of 2: commented out as this is a prediction problem
            #     return tape.gradient(probs, input_xs)
                return tape.gradient(logits, input_xs)

            def interpolate_input_xs(baseline,
                                    input_x,
                                    alphas):
                alphas_x = alphas[:, tf.newaxis, tf.newaxis] # change 1 of 2: Reduced 1 dimension to match the lstm input
                baseline_x = tf.expand_dims(baseline, axis=0)
                input_x = tf.expand_dims(input_x, axis=0)
                delta = input_x - baseline_x
                input_xs = baseline_x +  alphas_x * delta
                return input_xs

            @tf.function
            def one_batch(baseline, input_x, alpha_batch, target_class_idx):
                # Generate interpolated inputs between baseline and input.
                interpolated_path_input_batch = interpolate_input_xs(baseline=baseline,
                                                                    input_x=input_x,
                                                                    alphas=alpha_batch)

                # Compute gradients between model outputs and interpolated inputs.
                gradient_batch = compute_gradients(input_xs=interpolated_path_input_batch,
                                                    target_class_idx=target_class_idx)
                return gradient_batch

            def integral_approximation(gradients):
                # riemann_trapezoidal
                grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
                integrated_gradients = tf.math.reduce_mean(grads, axis=0)
                return integrated_gradients

            def integrated_gradients(baseline,
                                      input_x,
                                      target_class_idx,
                                      m_steps=50,
                                      batch_size=32):
                # Generate alphas.
                alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

                # Collect gradients.    
                gradient_batches = []

                # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
                for alpha in tf.range(0, len(alphas), batch_size):
                    from_ = alpha
                    to = tf.minimum(from_ + batch_size, len(alphas))
                    alpha_batch = alphas[from_:to]

                    gradient_batch = one_batch(baseline, input_x, alpha_batch, target_class_idx)
                    gradient_batches.append(gradient_batch)

                # Concatenate path gradients together row-wise into single tensor.
                total_gradients = tf.concat(gradient_batches, axis=0)

                # Integral approximation through averaging gradients.
                avg_gradients = integral_approximation(gradients=total_gradients)

                # Scale integrated gradients with respect to input.
                integrated_gradients = (input_x - baseline) * avg_gradients

                return integrated_gradients

            # LOAD THE MODEL EXTRACT EMBEDDINGS

            model_embeddings = tf.keras.Sequential()
            model_embeddings.add(tf.keras.layers.LSTM(100, activation = 'relu', return_sequences = True, input_shape = (input.shape[1], input.shape[2])))
            model_embeddings.add(tf.keras.layers.LSTM(32, activation = 'relu', return_sequences=True))
            model_embeddings.add(tf.keras.layers.Dense(1))#, activation='linear'
            model_embeddings.set_weights(model.get_weights()) # <- WE NEED TO CHANGE THIS

            # EXTRACTION
            intermediate_layer_model = keras.Model(inputs=model_embeddings.input,
                                                    outputs=model_embeddings.get_layer(model_embeddings.layers[len(model_embeddings.layers) - 1].name).output)

            intermediate_output = intermediate_layer_model(input)
            query_output = intermediate_layer_model(np.array(norm_instance).reshape(1,k,len(input_features))) 

            # GET MODEL PREDICTIONS
            predictions = model.predict(input)
            pred_list = []
            for pred in predictions.tolist():
              pred_list.append(pred[0])

            query_pred = model.predict(np.array(norm_instance).reshape(1,k,len(input_features)))[0][0]

            # GETTING THE EXPLANATION 
            input = np.float32(input)
            attributions_ig = []
            for input_x in input:
                attributions_ig.append(integrated_gradients(baseline=np.zeros(input.shape[1:], dtype='float32'),
                                                            input_x=input_x,
                                                            target_class_idx=0,
                                                            m_steps=10))
            attributions_ig = tf.stack(attributions_ig, axis=0, name='stack')

        
            intermediate_flattened = np.array(intermediate_output).reshape(len(intermediate_output), intermediate_output.shape[1] * intermediate_output.shape[2])
            query_flattened = np.array(query_output).reshape(len(query_output), query_output.shape[1] * query_output.shape[2])

            nbrs = NearestNeighbors(n_neighbors=NUM_NEIGHBOURS).fit(intermediate_flattened)
            distances, indices = nbrs.kneighbors(query_flattened)

            distances = np.squeeze(distances)
            indices = np.squeeze(indices)

            neighbours = []
            neighbour_features = []
            for i in range(NUM_NEIGHBOURS):
                NEIGHBOUR = {'score__': distances[i], 'index': indices[i]}
                FEATURES = np.array(data_values)[indices[i]]
                neighbours.append(NEIGHBOUR)
                neighbour_features.append(FEATURES)


            # GENERATE PARALLEL PLOTS

            plotdata_raw = []

            temp_dict = {}
            for i in range(0,k):
                temp_dict['Time:'+str(i+1)]=norm_instance[i][col]
            temp_dict['Relation']='Self'
            # temp_dict['Similarity']=col0['score__'][row_index]
            temp_dict['Time:'+str(k+1)+' Pred for ' +str(target_names[0])]=query_pred
            plotdata_raw.append(temp_dict)
        
            j=1
            for n in neighbours:
                n1_index = n['index']
                temp_dict = {}
                for i in range(0,k):
                    temp_dict['Time:'+str(i+1)]=input[n1_index,i,col]
                temp_dict['Relation']='Neighbor ' + str(j)
                # temp_dict['Similarity']=col1['score__'][row_index]
                temp_dict['Time:'+str(k+1)+' Pred for ' +str(target_names[0])]=pred_list[n1_index]
                plotdata_raw.append(temp_dict)
                j=j+1

            plotdata_raw = pd.DataFrame(plotdata_raw)

            map_dict={'Self': 1}
            j=1
            for n in neighbours:
                map_dict['Neighbor ' + str(j)]=j+1
                j=j+1

            plotdata_raw['Relation'] = plotdata_raw['Relation'].map(map_dict)
            fig = px.parallel_coordinates(plotdata_raw, color='Relation',
                                  dimensions=plotdata_raw.columns
                                 ,color_continuous_scale=px.colors.diverging.Tealrose
                                 ,color_continuous_midpoint=len(map_dict)/2
                                 ,title='Explanation for ' + str(feature_names[col]) + ': Relation=1 represents the input; Relation=2,3,4... represent the nearest neighbours'
                                 )
        
            #saving
            img_buf = BytesIO()
            fig.write_image(img_buf,width=k*150,height=500)
            im = Image.open(img_buf)
            b64Image=PIL_to_base64(im)

            response={"type":"image","explanation":b64Image}#,"explanation":dict_exp}
            return response
        except:
            return traceback.format_exc(), 500        

    def get(self,id=None):
        base_dict={
        "_method_description": "This method accepts 3 arguments: " 
                           "the 'id', the 'instance',  and the 'params' dictionary (optional) with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array fo dictionaries containing data for each time point in a single window.",
        "params": { 
                "num_neigbours" :{
                    "description":"Integer representing the number of neighbours to be included in the explanation. Defaults to 3.",
                    "type":"int",
                    "default":3,
                    "range":None,
                    "required":False
                    },
                "feature":{
                    "description":"Name of the feature to be included in the explanation. Defaults to the first feature in the dataset.",
                    "type":"string",
                    "default":None,
                    "range":None,
                    "required":False
                    }
                },
        "meta":{
                "modelAccess":"File",
                "supportsBWImage":False,
                "needsTrainingData": True
            }

        }

        if id is not None:
            #Getting model info, data, and file from local repository
            try:
                _, model_info_file, data_file = get_model_files(id,self.model_folder)
            except:
                return base_dict

            dataframe = pd.read_csv(data_file,header=0)
            model_info=json.load(model_info_file)
            target_names=model_info["attributes"]["target_names"]
            features=model_info["attributes"]["features"]
            feature_names=list(dataframe.columns)

            time_feature=None
            for feature, info_feature in features.items():
                if(info_feature["data_type"]=="time"):
                    time_feature=feature
                    break 

            if time_feature is not None:
                dataframe.drop([time_feature], axis=1,inplace=True)
                feature_names.remove(time_feature)
        
            for target in target_names:
                feature_names.remove(target)

            base_dict["params"]["feature"]["default"]=feature_names[0]
            base_dict["params"]["feature"]["range"]=feature_names

            return base_dict
           

        else:
            return base_dict

        
