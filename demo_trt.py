#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class ISNetDemo:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def forward(self, img):
        h, w = img.shape[:2]
        img = cv2.resize(img, (1024, 1024), cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0 
        img[:,:,0] = (img[:,:,0]-0.5)/1
        img[:,:,1] = (img[:,:,1]-0.5)/1
        img[:,:,2] = (img[:,:,2]-0.5)/1
        img = np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1))
        img = np.ascontiguousarray(img)
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        outputs = []
        for out in self.outputs:
            output = np.zeros(out['shape'],out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            outputs.append(output)
        output = outputs[0].squeeze()
        output = output * 255
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return output

if __name__ == "__main__":
    isnet = ISNetDemo('isnet.engine')
    image = cv2.imread('test.jpg')
    output = isnet.forward(image)
    cv2.imwrite('output_trt.png', output)
