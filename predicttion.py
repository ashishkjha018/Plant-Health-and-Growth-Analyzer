import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def preprocess_image(img_data):
    # Function to preprocess image data if needed
    # Example: Convert base64 string to image array
    image = Image.open(BytesIO(base64.b64decode(img_data)))
    # Additional preprocessing steps if required
    return image

def Apple(img_data):
    model = load_model('models/apple_vgg16.h5')
    class_names = ["scab","rot","Healthy"]
    
    # Preprocess image data
    img_array = preprocess_image(img_data)
    
    # Resize image if needed
    img_array = img_array.resize((224, 224))
    
    img_array = np.expand_dims(img_array, 0)
    vgg = model.predict(img_array)
    predicted_class = class_names[np.argmax(vgg[0])]
    return predicted_class

# Similar changes for other fruits

def Cherry(img_data):
    model = load_model('models/cherry_vgg16.h5')
    class_names = ["mildew","Healthy"]
    
    img_array = preprocess_image(img_data)
    img_array = img_array.resize((224, 224))
    
    img_array = np.expand_dims(img_array, 0)
    vgg = model.predict(img_array)
    predicted_class = class_names[np.argmax(vgg[0])]
    return predicted_class

# Update similar functions for other fruits

def Grape(img_data):
    model = load_model('models/grape_vgg16.h5')
    class_names = ["Measels","Leaf Blight","Healthy"]
    
    img_array = preprocess_image(img_data)
    img_array = img_array.resize((224, 224))
    
    img_array = np.expand_dims(img_array, 0)
    vgg = model.predict(img_array)
    predicted_class = class_names[np.argmax(vgg[0])]
    return predicted_class

def Strawberry(img_data):
    model = load_model('models/strawberry_vgg16.h5')
    class_names = ["Leaf Scorch","Healthy"]
    
    img_array = preprocess_image(img_data)
    img_array = img_array.resize((224, 224))
    
    img_array = np.expand_dims(img_array, 0)
    vgg = model.predict(img_array)
    predicted_class = class_names[np.argmax(vgg[0])]
    return predicted_class

def Potato(img_data):
    model = load_model('models/Potato_vgg16.h5')
    class_names = ["Early Blight","Late Blight","Healthy"]
    
    img_array = preprocess_image(img_data)
    img_array = img_array.resize((224, 224))
    
    img_array = np.expand_dims(img_array, 0)
    vgg = model.predict(img_array)
    predicted_class = class_names[np.argmax(vgg[0])]
    return predicted_class
