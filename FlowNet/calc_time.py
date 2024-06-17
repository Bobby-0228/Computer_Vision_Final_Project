import torch
import torch.nn.functional as F
import cv2

import torchvision.transforms as transforms
import flow_transforms
import numpy as np
from util import flow2rgb

import torch.backends.cudnn as cudnn
from time import time
import models

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


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cudnn.benchmark = True
    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )

    network_data = torch.load("/home/cv/Projects/Computer_Vision_Final_Project/FlowNet/trained_model/model_c_best.pth.tar")
    model = models.__dict__[network_data["arch"]](network_data).to(device)
    model.eval()

    if "div_flow" in network_data.keys():
        # what is this??
        div_flow = network_data["div_flow"]
    else:
        div_flow = 20.0

    def detect_movement(frame1, frame2, sensitivity=30):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
        num_white_pixels = cv2.countNonZero(thresh)
        movement_threshold = 1000
        return num_white_pixels > movement_threshold



    cap = cv2.VideoCapture("/home/cv/Projects/Computer_Vision_Final_Project/FlowNet/gifs/IMG_1667.mov")
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    flow = calcOpticalFlowFlownet(model, frame1, frame2, div_flow, upsampling=None)

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    total_time = 0
    count = 0
    while(1):
        time_stemp = time()

        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break


        # if detect_movement(frame1, frame2):
        #     flow_s = calcOpticalFlowFlownet(model, frame1, frame2, div_flow, upsampling=None)
        # else:
        #     flow_s = np.ones_like(flow_s, dtype=np.float32)


        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_farneback = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prvs = next

        frame1 = frame2

        count += 1
        total_time += time()-time_stemp

    print(total_time/count)

