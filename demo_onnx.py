import os
import sys
import cv2
import numpy as np
import timeit
import onnxruntime
class ISNetDemo:
    def __init__(self, model_path):
        self.ort_session = onnxruntime.InferenceSession(model_path)
    def forward(self, img):
        h, w = img.shape[:2]
        img = cv2.resize(img, (1024, 1024), cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0 
        img[:,:,0] = (img[:,:,0]-0.5)/1
        img[:,:,1] = (img[:,:,1]-0.5)/1
        img[:,:,2] = (img[:,:,2]-0.5)/1
        img = np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1))
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0].squeeze()
        output = output * 255
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return output

if __name__ == "__main__":
    isnet = ISNetDemo('isnet.onnx')
    img = cv2.imread('test.jpg')
    output = isnet.forward(img)
    cv2.imwrite('output_onnx.png', output)
