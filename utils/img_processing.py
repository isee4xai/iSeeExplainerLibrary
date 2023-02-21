import numpy as np

def denormalize_img(instance,model_info):
    if("min" in model_info["attributes"]["features"]["image"] and "max" in model_info["attributes"]["features"]["image"] and
        "min_raw" in model_info["attributes"]["features"]["image"] and "max_raw" in model_info["attributes"]["features"]["image"]):
        nmin=model_info["attributes"]["features"]["image"]["min"]
        nmax=model_info["attributes"]["features"]["image"]["max"]
        min_raw=model_info["attributes"]["features"]["image"]["min_raw"]
        max_raw=model_info["attributes"]["features"]["image"]["max_raw"]
        try:
            instance=(((instance-nmin)/(nmax-nmin))*(max_raw-min_raw)+min_raw).astype(np.uint8)
        except Exception as e:
            raise
    elif("mean_raw" in model_info["attributes"]["features"]["image"] and "std_raw" in model_info["attributes"]["features"]["image"]):
        mean=np.array(model_info["attributes"]["features"]["image"]["mean_raw"])
        std=np.array(model_info["attributes"]["features"]["image"]["std_raw"])
        try:
            instance=((instance*std)+mean).astype(np.uint8)
        except Exception as e:
            raise

    if instance.shape!=tuple(model_info["attributes"]["features"]["image"]["shape_raw"]):
        try:
            instance=instance.reshape(tuple(model_info["attributes"]["features"]["image"]["shape_raw"]))
        except Exception as e:
            print(e)
            return "Cannot reshape image of shape " + str(instance.shape) + " into shape " + str(tuple(model_info["attributes"]["features"]["image"]["shape_raw"]))
    return instance

def normalize_img(instance,model_info):
    if("min" in model_info["attributes"]["features"]["image"] and "max" in model_info["attributes"]["features"]["image"] and
        "min_raw" in model_info["attributes"]["features"]["image"] and "max_raw" in model_info["attributes"]["features"]["image"]):
        nmin=model_info["attributes"]["features"]["image"]["min"]
        nmax=model_info["attributes"]["features"]["image"]["max"]
        min_raw=model_info["attributes"]["features"]["image"]["min_raw"]
        max_raw=model_info["attributes"]["features"]["image"]["max_raw"]
        try:
            instance=((instance-min_raw) / (max_raw - min_raw)) * (nmax - nmin) + nmin
        except Exception as e:
            raise
    elif("mean_raw" in model_info["attributes"]["features"]["image"] and "std_raw" in model_info["attributes"]["features"]["image"]):
        mean=np.array(model_info["attributes"]["features"]["image"]["mean_raw"])
        std=np.array(model_info["attributes"]["features"]["image"]["std_raw"])
        try:
            instance=((instance-mean)/std).astype(np.uint8)
        except Exception as e:
            raise

    if instance.shape!=tuple(model_info["attributes"]["features"]["image"]["shape"]):
        try:
            instance = instance.reshape(tuple(model_info["attributes"]["features"]["image"]["shape"]))
        except Exception as e:
            print(e)
            return "Cannot reshape image of shape " + str(instance.shape) + " into shape " + str(tuple(model_info["attributes"]["features"]["image"]["shape"]))
    instance=instance.reshape((1,)+instance.shape)
    return instance