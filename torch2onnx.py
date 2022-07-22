# -*- coding: utf-8 -*-
#import cv2
import numpy as np
import time
import torch
import pdb
from collections import OrderedDict
import sys
import onnxruntime
from isnet import *

net = ISNetDIS()
net.load_state_dict(torch.load('./isnet.pth', map_location=torch.device('cpu')))
input = torch.randn(1, 3, 1024, 1024, device='cpu')
torch.onnx.export(net, input, 'isnet.onnx',
                export_params=True, opset_version=11, do_constant_folding=True,
                input_names = ['input'])

