from flask import Flask, request, jsonify, flash, redirect, url_for
from flask import render_template
app = Flask(__name__)
import io

import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.utils import save_image
import json
import requests

import os
from werkzeug.utils import secure_filename

"""
================================================================================
set directories, load model
================================================================================
"""
cwd = os.getcwd()
UPLOAD_FOLDER = os.path.join(cwd,'static')
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

imagenet_class_index = json.load(open('imagenet_class_index.json'))
# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()

"""
================================================================================
define functions for model and file preps
================================================================================
"""
def get_prediction(image_bytes, filename):
    tensor = transform_image(image_bytes=image_bytes, filename=filename)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)   # alternatively, can spit out more predictions
                                # it will also have an option for assigning 
                                # probs, but I'm not sure how reliable those are
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

# transform image so it can be fed into model: its a bit ugly as it just
# crops the image, but it will do for now
def transform_image(image_bytes, filename):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    # save image crop into static for display
    transf = my_transforms(image)
    save_image(transf, 'static/'+str(filename)+'tf.jpg')
    return transf.unsqueeze(0)

# check for valid files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
================================================================================
set up routes
================================================================================
"""
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)
            # I'm just calling a different endpoint for eval, i think this might
            # be pretty sloppy
            class_id = requests.post("http://localhost:5000/predict",
                     files={"file": open('static/'+filename,'rb')})
            print(class_id.content)
            return render_template('output.html', 
                                response=class_id.content,
                                filename='./static/'+filename[:-4]+'tf.jpg')

    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        _, class_name = get_prediction(image_bytes=img_bytes, 
                                       filename=file.filename[:-4])
        return 'class_name: '+class_name

"""
================================================================================
run
================================================================================
"""        
if __name__ == '__main__':
    app.run()