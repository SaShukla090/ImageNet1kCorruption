import os
from utils import validate, CustomImageDataset, buildmodel
from imagecorruptions import corrupt, get_corruption_names
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
import pandas as pd
dataset = load_dataset("imagenet-1k")
model_names = [
'vgg16bn',
'vgg19bn',
'resnet18',
'resnet50',
'mobilenet_v2',
'googlenet',
'densenet121',
'densenet161',
'shufflenet_v2_x0_5',
'shufflenet_v2_x2_0'
]

path = '/home/tbvl/MS_Students/Shivam/cifar_testing_chapter0/results'
for model_name in model_names:
    model = buildmodel(model_name)
    df_result_top1 = pd.DataFrame(columns=get_corruption_names(), index=[f"S{i}" for i in range(1, 6)])
    df_result_top5 = pd.DataFrame(columns=get_corruption_names(), index=[f"S{i}" for i in range(1, 6)])
    for corruption_name in get_corruption_names():
        for severity in range(1, 6):
            print(model_name,corruption_name, severity)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            val_dataset = CustomImageDataset(dataset['validation'], corruption_name=corruption_name, severity=severity)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=64)
            top1, top5 = validate(model,val_loader, device)
            df_result_top1.loc[f"S{severity}", corruption_name] = top1
            df_result_top5.loc[f"S{severity}", corruption_name] = top5
            df_result_top1.to_csv(os.path.join(path,f'df_result_top1_{model_name}.csv'), index=False)
            df_result_top5.to_csv(os.path.join(path,f'df_result_top5_{model_name}.csv'), index=False)
            del val_loader
    del model