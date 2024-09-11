# @author : Pratham Shinde, Abhishek Ajatdesai, Om kangralkar
# @version : v1.2.0
# @date : 12-June-2024


import os
import sys
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

# Add the absolute path to the directory containing the 'net' package
sys.path.append('/absolute/path/to/your/project')

from net.Ushape_Trans import Generator

app = Flask(__name__)

# Load the pre-trained model
generator = Generator()
generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth", map_location=torch.device('cpu')))
generator.eval()

UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './test/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def process_image(input_path, output_path):
    imgx = cv2.imread(input_path)
    imgx = cv2.resize(imgx, (256, 256))
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = np.array(imgx).astype('float32')
    imgx = torch.from_numpy(imgx)
    imgx = imgx.permute(2, 0, 1).unsqueeze(0)
    imgx = imgx / 255.0
    imgx = Variable(imgx)
    output = generator(imgx)
    out = output[3].data
    save_image(out, output_path, nrow=5, normalize=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
        file.save(input_path)
        process_image(input_path, output_path)
        return render_template('result.html', filename=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
