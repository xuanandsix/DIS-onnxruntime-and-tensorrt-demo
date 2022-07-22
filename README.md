# DIS-onnxruntime-and-tensorrt-demo
This is the onnxruntime and tensorrt inference code  for Highly Accurate Dichotomous Image Segmentation (ECCV 2022).
Official code: https://github.com/xuebinqin/DIS

## test onnx
1、cp pre-train model isnet.pth to here. <br>
2、run python torch2onnx.py, get model isnet.onnx <br>
3、run python demo_onnx.py, get image output.

## test tensorrt
1、Use trtexec tool convert onnx model to trt model. You can also try something else, please make sure to get the correct trt model.
  Name it isnet.engine. <br>
2、run python demo_trt.py, get image output. <br>

## TO DO
- [ ] c++ code for tensorrt

## output compare

###
| input | pytorch| onnx | tensorrt|
| :-: |:-:| :-:|:-:|
|<img src="https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo/raw/main/test.jpg" >|<img src="https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_torch.png">|<img src="https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_onnx.png" >|<img src="https://github.com/xuanandsix/DIS-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_trt.png">|
