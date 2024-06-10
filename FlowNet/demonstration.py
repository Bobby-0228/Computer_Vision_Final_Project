import torch
import cv2 as cv
import numpy as np
import models
import torch.backends.cudnn as cudnn

from imageio.v2 import imread, imwrite
from flow_estimate import calcOpticalFlowFlownet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cudnn.benchmark = True
model_names = sorted(
    name for name in models.__dict__ if name.islower() and not name.startswith("__")
)

network_s_data = torch.load("/home/cv/Projects/FlowNetPytorch/copy/model_s_best.pth.tar")
model_s = models.__dict__[network_s_data["arch"]](network_s_data).to(device)
model_s.eval()

network_c_data = torch.load("/home/cv/Projects/FlowNetPytorch/copy/model_c_best.pth.tar")
model_c = models.__dict__[network_c_data["arch"]](network_c_data).to(device)
model_c.eval()

if "div_flow" in network_s_data.keys():
    # what is this??
    div_flow = network_s_data["div_flow"]
else:
    div_flow = 20.0

cap = cv.VideoCapture(0)
ret, frame1 = cap.read()
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    flow = calcOpticalFlowFlownet(model_s, frame1, frame2, div_flow, upsampling=None)
    flow = calcOpticalFlowFlownet(model_c, frame1, frame2, div_flow, upsampling=None)
    cv.imshow("Optical Flow by FlowNet_s", cv.cvtColor(flow, cv.COLOR_RGB2BGR) )
    cv.imshow("Optical Flow by FlowNet_c", cv.cvtColor(flow, cv.COLOR_RGB2BGR) )
    cv.imshow("Camera Sight", frame2)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    frame1 = frame2
cv.destroyAllWindows()