
from PIL import Image
import io
import json
import logging
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#=======================================================================
#                        MODEL DEFINITION
#=======================================================================

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(-1, 256)
        
        x = F.leaky_relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.leaky_relu(self.bn6(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
    
#=======================================================================
#                       PREPARE MODEL
#=======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
def model_fn(model_dir):
    # nn.DataParallel() wrapping is because the model training weights were saved w/ 'module.' prefix
    model = CIFAR10Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model

#=======================================================================
#                      INFERENCE FUNCTIONS
#=======================================================================

# Deserialize & pre-process input data
def input_fn(request_body, request_content_type):
    if request_content_type == "application/x-image":
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        image = Image.open(io.BytesIO(request_body))
        return img_transforms(image).unsqueeze(0)
    raise ValueError("Unsupported content type: {}".format(request_content_type))

# Perform prediction
def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
        probabilities = F.softmax(output, dim=1)
    return probabilities

# Post-process the prediction
def output_fn(predictions, content_type):
    # assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)
