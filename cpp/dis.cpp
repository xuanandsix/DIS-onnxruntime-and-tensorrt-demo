#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logging.h"

using namespace sample;

void preprocess(cv::Mat& img, float data[]) {
    if (img.type() != CV_32FC3) {
        img.convertTo(img, CV_32FC3);
    }
	cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
    cv::resize(img, img, cv::Size(1024,1024), 0, 0, cv::INTER_LINEAR);
    img = img / 255.0;
    cv::divide(img, cv::Scalar(1.0, 1.0, 1.0), img);
    cv::subtract(img, cv::Scalar(0.5, 0.5, 0.5), img);

    int channelLength = 1024 * 1024; 
    int index = 0;
    std::vector<cv::Mat> chs = {
      cv::Mat(1024, 1024, CV_32FC1, data + channelLength * (index + 0)),
      cv::Mat(1024, 1024, CV_32FC1, data + channelLength * (index + 1)),
      cv::Mat(1024, 1024, CV_32FC1, data + channelLength * (index + 2))
    };
    cv::split(img, chs);
}

// detect
void detect( cv::Mat &output, float* output_tensor) {
    cv::Mat prob_mat = cv::Mat_<float>(1024, 1024);
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            prob_mat.at<float>(i, j) = output_tensor[i * 1024 + j] *255;
        }
    }
    prob_mat.convertTo(output, CV_8UC1);
}


float input_tensor[1024 * 1024 * 3]; 
float output_tensor[1024 * 1024];

int main()
{
    Logger gLogger;
    nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
    std::string engine_filepath = "../../isnet.engine";
    std::ifstream file;
    file.open(engine_filepath, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    int length = file.tellg();
    file.seekg(0, std::ios::beg);
    std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
    file.read(data.get(), length);
    file.close();
    nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);
    nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();
    int input_index = engine_infer->getBindingIndex("input"); 
    int output_index_1 = engine_infer->getBindingIndex("1877");
    if (engine_context == nullptr)
    {
        std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
    }
	cv::Mat image = cv::imread("../../test.jpg");
	cv::Mat source_img = image.clone();
	preprocess(image, input_tensor);
	void* buffers[2];
	cudaMalloc(&buffers[0], 1024 * 1024 * 3 * sizeof(float));  
	cudaMalloc(&buffers[1], 1024 * 1024 * sizeof(float));
	cudaMemcpy(buffers[0], input_tensor, 1024 * 1024 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	engine_context->executeV2(buffers);
	cudaMemcpy(output_tensor, buffers[1], 1024 * 1024 * sizeof(float), cudaMemcpyDeviceToHost);
	cv::Mat output;
	detect(output, output_tensor);
	cv::resize(output, output, cv::Size(source_img.size[1], source_img.size[0]), 0, 0, cv::INTER_LINEAR);
	cv::imwrite("output_cpp_trt.png", output);
	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
    
    engine_runtime->destroy();
    engine_infer->destroy();

    return 0;
}

