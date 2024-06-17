import torch
import torch.nn.functional as F
import cv2

import torchvision.transforms as transforms
import flow_transforms
import numpy as np
from util import flow2rgb

from time import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def calcOpticalFlowFlownet(model, img1, img2, div_flow, max_flow=None, upsampling=None):
    input_transform = transforms.Compose(
        [
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
        ]
    )
    

    img1 = input_transform(img1)
    img2 = input_transform(img2)
    input_var = torch.cat([img1, img2]).unsqueeze(0)

    input_var = input_var.to(device)
    # compute output
    
    output = model(input_var)[0]

    if upsampling is not None:
        output = F.interpolate(
            output, size=img1.size()[-2:], mode=upsampling, align_corners=False
        )
    rgb_flow = flow2rgb(
        div_flow * output, max_value=max_flow
    )

    return (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)