import numpy as np

def denormalize_dataframe(df,model_info):
    denorm_df=df.copy()
    column_names=list(denorm_df.columns)
    for feature in column_names:
        feature_dict=model_info["attributes"]["features"][feature]
        if(feature_dict["data_type"]=="numerical"):
            if("min" in feature_dict and "max" in feature_dict and "min_raw" in feature_dict and "max_raw" in feature_dict):
                nmin=feature_dict["min"]
                nmax=feature_dict["max"]
                min_raw=feature_dict["min_raw"]
                max_raw=feature_dict["max_raw"]
                try:
                    denorm_df[feature]=(((denorm_df[feature]-nmin)/(nmax-nmin))*(max_raw-min_raw)+min_raw)
                except:
                    raise
            elif ("mean_raw" in feature_dict and "std_raw" in feature_dict):
                mean=np.array(feature_dict["mean_raw"])
                std=np.array(feature_dict["std_raw"])
                try:
                    denorm_df[feature]=((denorm_df[feature]*std)+mean)
                except Exception as e:
                    raise
        elif feature_dict["data_type"]=="categorical":
            if("values" in feature_dict and "values_raw" in feature_dict):
                try:
                    denorm_df[feature]=denorm_df[feature].apply(lambda row: feature_dict["values_raw"][int(row)])
                except:
                    pass
            elif("value" in feature_dict and "ohe_feature" in feature_dict):
                if(denorm_df[feature].values[0]==0):
                    denorm_df=denorm_df.drop(feature,axis=1)
                else:
                    denorm_df[feature]=feature_dict["value"]
                    denorm_df=denorm_df.rename(columns={feature: feature_dict["ohe_feature"]})
                                        
    return denorm_df

def normalize_dataframe(df,model_info):
    norm_df=df.copy()
    column_names=list(norm_df.columns)
    for feature in column_names:
        feature_dict=model_info["attributes"]["features"][feature]
        if(feature_dict["data_type"]=="numerical"):
            if("min" in feature_dict and "max" in feature_dict and "min_raw" in feature_dict and "max_raw" in feature_dict):
                nmin=feature_dict["min"]
                nmax=feature_dict["max"]
                min_raw=feature_dict["min_raw"]
                max_raw=feature_dict["max_raw"]
                try:
                    norm_df[feature]=((norm_df[feature]-min_raw) / (max_raw - min_raw)) * (nmax - nmin) + nmin
                except:
                    raise
            elif ("mean_raw" in feature_dict and "std_raw" in feature_dict):
                mean=np.array(feature_dict["mean_raw"])
                std=np.array(feature_dict["std_raw"])
                try:
                    norm_df[feature]=((norm_df[feature]-mean)/std)
                except Exception as e:
                    raise
        elif feature_dict["data_type"]=="categorical":
            if("values" in feature_dict and "values_raw" in feature_dict):
                try:
                    norm_df[feature]=norm_df[feature].apply(lambda row: feature_dict["values"][feature_dict["values_raw"].index(row)])               
                except:
                    raise
            elif("value" in feature_dict and "ohe_feature" in feature_dict):
                if(norm_df[feature].values[0]==0):
                    norm_df=norm_df.drop(feature,axis=1)
                else:
                    norm_df[feature]=feature_dict["value"]
                    norm_df=norm_df.rename(columns={feature: feature_dict["ohe_feature"]})    
                                        
    return norm_df

def normalize_dict(instance,model_info):
    dictionary=instance.copy()
    column_names=list(dictionary.keys())
    for feature in column_names:
        feature_dict=model_info["attributes"]["features"][feature]
        if(feature_dict["data_type"]=="numerical"):
            if("min" in feature_dict and "max" in feature_dict and "min_raw" in feature_dict and "max_raw" in feature_dict):
                nmin=feature_dict["min"]
                nmax=feature_dict["max"]
                min_raw=feature_dict["min_raw"]
                max_raw=feature_dict["max_raw"]
                try:
                    dictionary[feature]=((dictionary[feature]-min_raw) / (max_raw - min_raw)) * (nmax - nmin) + nmin
                except:
                    raise
            elif ("mean_raw" in feature_dict and "std_raw" in feature_dict):
                mean=np.array(feature_dict["mean_raw"])
                std=np.array(feature_dict["std_raw"])
                try:
                    dictionary[feature]=((dictionary[feature]-mean)/std)
                except Exception as e:
                    raise
        elif feature_dict["data_type"]=="categorical":
            if("values" in feature_dict and "values_raw" in feature_dict):
                try:
                    dictionary[feature]=feature_dict["values"][feature_dict["values_raw"].index(dictionary[feature])]               
                except:
                    raise
            elif("value" in feature_dict and "ohe_feature" in feature_dict):
                if(dictionary[feature].values[0]==0):
                    dictionary=dictionary.drop(feature,axis=1)
                else:
                    dictionary[feature]=feature_dict["value"]
                    dictionary=dictionary.rename(columns={feature: feature_dict["ohe_feature"]})                           
    return dictionary

def split_sequences(sequences, n_steps):
  input = list()
  for i in range(len(sequences)):
		# find the end of this pattern
    end_ix = i + n_steps
		# check if we are beyond the dataset
    if end_ix > len(sequences):
      break
		# gather input and output parts of the pattern
    seq = sequences[i:end_ix] #sequences[i:end_ix,:-1], sequences[end_ix:end_ix+1,-1]
    input.append(seq)
  return np.array(input)