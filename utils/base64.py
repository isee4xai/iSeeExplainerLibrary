import base64
from io import BytesIO
from PIL import Image
import numpy as np

def vector_to_base64(image):
    image=Image.fromarray(np.squeeze(image))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")

def base64_to_vector(b64_str):
    img=b64_str.encode("utf-8")
    image_bytes=base64.b64decode(img)
    img=Image.open(BytesIO(image_bytes))
    return np.asarray(img)

def PIL_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")