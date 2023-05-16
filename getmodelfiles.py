from os.path import exists



def get_model_files(_id,model_folder):

    if exists(model_folder+'/' +_id):
        path=model_folder+'/' +_id+'/'+_id
    
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
        elif exists(path + '_data.csv'):
            data=open(path + '_data.csv','r')
        elif exists(path+"_data"):
            data=path+"_data"

        return model, model_info, data
    else:
        raise Exception("No directory with id '"+ _id +"' was found in the database.")
