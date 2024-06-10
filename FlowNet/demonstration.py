import torch
import cv2
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

network_sref_data = torch.load("/home/cv/Projects/FlowNetPytorch/copy/model_s_best.pth.tar")
model_sref = models.__dict__[network_sref_data["arch"]](network_sref_data).to(device)
model_sref.eval()

network_cref_data = torch.load("/home/cv/Projects/FlowNetPytorch/copy/model_c_best.pth.tar")
model_cref = models.__dict__[network_cref_data["arch"]](network_cref_data).to(device)
model_cref.eval()

if "div_flow" in network_s_data.keys():
    # what is this??
    div_flow = network_s_data["div_flow"]
else:
    div_flow = 20.0

def detect_movement(frame1, frame2, sensitivity=30):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference between frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold to detect significant changes
    _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
    
    # Count the number of white pixels in the thresholded image
    num_white_pixels = cv2.countNonZero(thresh)
    
    # Define a threshold for the number of white pixels to determine movement
    movement_threshold = 1000
    
    # Return True if movement is detected, False otherwise
    return num_white_pixels > movement_threshold



cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
flow_s = calcOpticalFlowFlownet(model_s, frame1, frame2, div_flow, upsampling=None)
while(1):
    ret, frame2 = cap.read()

    if detect_movement(frame1, frame2):
        flow_s = calcOpticalFlowFlownet(model_s, frame1, frame2, div_flow, upsampling=None)
    else:
        flow_s = np.ones_like(flow_s, dtype=np.float32)
    # flow_c = calcOpticalFlowFlownet(model_c, frame1, frame2, div_flow, upsampling=None)
    cv2.imshow("Optical Flow by FlowNet_s", cv2.cvtColor(flow_s, cv2.COLOR_RGB2BGR) )
    # cv2.imshow("Optical Flow by FlowNet_c", cv2.cvtColor(flow_c, cv2.COLOR_RGB2BGR) )

    # flow_sref = calcOpticalFlowFlownet(model_sref, frame1, frame2, div_flow, upsampling=None)
    # flow_cref = calcOpticalFlowFlownet(model_cref, frame1, frame2, div_flow, upsampling=None)
    # cv2.imshow("Optical Flow by FlowNet_s_ref", cv2.cvtColor(flow_sref, cv2.COLOR_RGB2BGR) )
    # cv2.imshow("Optical Flow by FlowNet_c_ref", cv2.cvtColor(flow_cref, cv2.COLOR_RGB2BGR) )

    cv2.imshow("Camera Sight", frame2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    frame1 = frame2
cv2.destroyAllWindows()