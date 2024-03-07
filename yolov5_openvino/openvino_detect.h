#ifndef GMASTER_CV_2023_ARMORNEWYOLO_H
#define GMASTER_CV_2023_ARMORNEWYOLO_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <openvino/openvino.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
#define IMG_SIZE 640 // 推理图像大小,GMaster团队使用416x416作为图像推理大小
#define DEVICE "CPU" // 设备选择，Inter NUC13 应使用GNA，其他使用GPU
#define VIDEO        // 是否展示推理视频
#define MODEL_PATH "model/small/small.xml"
#define CLS_NUM 12
#define NMS_THRESHOLD 0.25
#define CONF_THRESHOLD 0.7
#define SCORE_THRESHOLD 0.3
#define MODEL_TYPE "yolov5"
#endif

class yolo_detct
{
public:
    yolo_detct();

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };
    struct Resize
    {
        cv::Mat resized_image;
        int dw;
        int dh;
    };

    std::vector<Object> work(cv::Mat src_img);
#ifdef MODEL_TYPE
    const std::vector<std::string> class_names = {
        "B1", "B2", "B3", "B4", "B5", "B7", "R1", "R2", "R3", "R4", "R5", "R7"};
#endif
#ifndef MODEL_TYPE
    const std::vector<std::string> class_names = {
        "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS"};
#endif

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;

    static float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }
};
