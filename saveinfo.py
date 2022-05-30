from flask import request
import random
import string

UPLOAD_FOLDER="Uploads/"

def save_file_info(name):
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + name.replace("/","_")
    return UPLOAD_FOLDER, filename, request.host_url + "ViewExplanation/" + filename