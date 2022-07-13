from flask import request
import random
import string



def save_file_info(name, upload_folder):
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + name.replace("/","_")
    return upload_folder+'/', filename, request.host_url + "ViewExplanation/" + filename