from os.path import exists
MODEL_FOLDER="Models/"

def get_model_files(_id):

    path=MODEL_FOLDER+_id+'/'+_id
    
    model=None
    if exists(path + '.pkl'):
        model = open(path + '.pkl','r+b')
    elif exists(path + '.h5'):
        model = open(path + '.h5','r+b')
    elif exists(path + '.pt'):
        model = open(path + '.pt','r+b')

    model_info = None
    if exists(path + '.json'):
        model_info=open(path + '.json')

    data=None
    if exists(path + '_data.pkl'):
        data=open(path + '_data.pkl','rb')

    return model, model_info, data
