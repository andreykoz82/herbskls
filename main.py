import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, make_response, request

CLASS_NAMES = os.listdir('data/dataset/train')
NUM_CLASSES = len(CLASS_NAMES)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnext50_32x4d(pretrained=True, progress=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model_ft.eval()
    return model_ft


def make_predictions(path_to_file, model):
    test_transform = transforms.Compose([
        transforms.CenterCrop(200),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    img = Image.open(path_to_file)
    x = test_transform(img)
    x = x.unsqueeze(0)
    output = model(x)
    prediction = torch.argmax(output, 1)
    return CLASS_NAMES[prediction]


model = load_model('models/resnext50_32x4d_gpu.pth')

app = Flask(__name__)


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>Определение вида растения</h1>
                </br>
                </br>
                <p> Загрузите фото растения в сухом виде
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="ОК" class="btn btn-primary btn-block btn-large">Определить</button>
                </form>
            </body>
        </html>
    """


@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"
    output = make_predictions(request.files['data_file'], model)
    response = make_response(output)
    return response


if __name__ == "__main__":
    app.run(debug=False)
