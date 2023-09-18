
#=======================================================================
#                               SET UP
#=======================================================================

# ensure that the latest version of the SageMaker SDK is available
import os

os.system("pip install -U sagemaker")

import argparse
import json
import logging
import traceback
import sys
import time
from os.path import join
import boto3
from sagemaker.session import Session
from sagemaker.experiments.run import load_run
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout)) # Sends logs to std out

# Saving copy of logs to .json file
if "SAGEMAKER_METRICS_DIRECTORY" in os.environ:
    log_file_handler = logging.FileHandler(
        join(os.environ["SAGEMAKER_METRICS_DIRECTORY"], "metrics.json")
    )
    formatter = logging.Formatter(
        "{'time':'%(asctime)s', 'name': '%(name)s', \
        'level': '%(levelname)s', 'message': '%(message)s'}",
        style="%",
    )
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)

#=======================================================================
#                               MODEL
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
#                               EVAL LOG FUNCTION
#=======================================================================


def log_performance(model, data_loader, device, epoch, run, metric_type="Test"):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss += torch.nn.functional.nll_loss(
            #     output, target, reduction="sum"
            # ).item()  # sum up batch loss
            criterion = nn.CrossEntropyLoss(reduction = "sum")
            loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    # ⛳️SM: log metrics
    run.log_metric(name=metric_type + ":loss", value=loss, step=epoch)
    run.log_metric(name=metric_type + ":accuracy", value=accuracy, step=epoch)
    logger.info(
        "{} Average loss: {:.4f}, {} Accuracy: {:.4f}%;\n".format(
            metric_type, loss, metric_type, accuracy
        )
    )

#=======================================================================
#                               TRAIN FUNCTION
#=======================================================================

def train_model(
    run, 
    train_set, 
    test_set,
    epochs, 
    lr
):
    """
    Args:
        run (sagemaker.experiments.run.Run): SageMaker Experiment run object
        train_set (torchvision.datasets.mnist.MNIST): train dataset
        test_set (torchvision.datasets.mnist.MNIST): test dataset
        data_dir (str): local directory where the MNIST datasource is stored
        optimizer (str): the optimization algorthm to use for training your CNN
                         available options are sgd and adam
        epochs (int): number of complete pass of the training dataset through the algorithm
        hidden_channels (int): number of hidden channels in your model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size= 64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size= 500, shuffle=False)
    
    #📍Logger: Progress Output
    logger.info(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),))

    logger.info(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),))
    
    # Train Set Up
    model = CIFAR10Net().to(device)
    # model = torch.nn.DataParallel(model) # if multiple GPU's
    lr = lr
    log_interval = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
        
    # ⛳️SM: log model run parameters
    run.log_parameters({"optimizer": "Adam",
                        "epochs": epochs,})

    # Train Loop = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    for epoch in range(1, epochs + 1):
        print("Training Epoch:", epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = torch.nn.functional.nll_loss(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            #📍 Logger: Train Status
            if batch_idx % log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.6f};".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        # ⛳️ SM: Metric Logging
        run.log_metric(name = 'epoch', value = epoch)
        log_performance(model, train_loader, device, epoch, run, "Train")
        log_performance(model, test_loader, device, epoch, run, "Test")
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        
    # ⛳️ SM: Confusion Matrix Logging
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            run.log_confusion_matrix(target, pred, "Confusion-Matrix-Test-Data")
    return model


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_channels = int(os.environ.get("hidden_channels", "5"))
    kernel_size = int(os.environ.get("kernel_size", "5"))
    dropout = float(os.environ.get("dropout", "0.5"))
    model = torch.nn.DataParallel(Net(hidden_channels, kernel_size, dropout))
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        return model.to(device)


def save_model(model, model_dir, run):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

#=======================================================================
#                               ARGPARSE
#=======================================================================

if __name__ == "__main__":
    
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("--epochs",type=int,default=10,metavar="N",help="number of epochs to train (default: 10)")
        parser.add_argument('--lr', type=float, default=0.01)

        # Data, model, and output directories
        # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
        parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
        # parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

        # Container environment
        parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
        parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
        parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
        parser.add_argument("--region", type=str, default="us-east-1", help="SageMaker Region")

        args, _ = parser.parse_known_args()

# = = = = = = = = = = = = = = DATA = = = = = = = = = = = = = = = = = = = =

        load_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = torchvision.datasets.ImageFolder(root=args.training, transform=load_transform)
        test_set = torchvision.datasets.ImageFolder(root=args.validation, transform=load_transform)

#=======================================================================
#                               RUN
#=======================================================================

        session = Session(boto3.session.Session(region_name=args.region))

        with load_run(sagemaker_session=session) as run:

            model = train_model(
                run,
                train_set = train_set,
                test_set = test_set,
                epochs=args.epochs,
                lr = args.lr
            )
            save_model(model, args.model_dir, run)
            
    except Exception as e:
        logger.error("Exception occurred: %s", e)
        logger.error(traceback.format_exc())
