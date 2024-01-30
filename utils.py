import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("imagenet-1k")

from imagecorruptions import corrupt

from PIL import Image
import numpy as np

# Load the image in grayscale mode
# Replace 'path_to_image' with your image's path
# gray_image = Image.open('path_to_image').convert('L')

# Convert 1-channel grayscale image to 3-channel
# gray_image_3_channel = np.stack((gray_image,)*3, axis=-1)
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained VGG16 model
# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, hf_dataset, corruption_name=None, severity=None):
        self.hf_dataset = hf_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        if self.corruption_name is not None:
            assert self.severity is not None
            self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Resize images
            torchvision.transforms.Lambda(lambda image: corrupt(np.array(image), corruption_name=corruption_name, severity=severity)),  # Convert grayscale to RGB
            torchvision.transforms.ToTensor(),  # Convert images to PyTorch tensors
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images
            # transforms.Lambda(lambda image: image.convert('RGB')),  # Convert grayscale to RGB
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Load image and label from Hugging Face dataset
        image = self.hf_dataset[idx]['image']
        label = self.hf_dataset[idx]['label']
        # print(type(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Transform image
        if self.transform:
            image = self.transform(image)
        
        return  image, label







from tqdm import tqdm
def validate(model, data_loader, device):
    model.eval()
    model.to(device)
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.topk(5, 1, largest=True, sorted=True)

            total_samples += targets.size(0)
            top1_correct += predicted[:, 0].eq(targets).sum().item()
            top5_correct += predicted.eq(targets.view(-1, 1)).sum().item()

    top1_accuracy = 100.0 * top1_correct / total_samples
    top5_accuracy = 100.0 * top5_correct / total_samples

    return top1_accuracy, top5_accuracy


def buildmodel(model_name):
    if model_name == 'vgg16bn':
        model = models.vgg11_bn(pretrained = True)
    elif model_name == 'vgg19bn':
        model = models.vgg19_bn(pretrained = True)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained = True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained = True)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained = True)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained = True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained = True)
    elif model_name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained = True)
    elif model_name == 'shufflenet_v2_x2_0':
        model = models.shufflenet_v2_x2_0(pretrained = True)
    return model
    

