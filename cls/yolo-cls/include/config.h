#pragma once
// 精度 建议别用INT8，因为需要校准集，我自己也没用过
// #define USE_FP16
 #define USE_FP32
// #define USE_INT8

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output";
const static char* kProtoTensorName = "proto";

// 模型种类数，需要根据实际的模型修改(for seg and det )
const static int kNumClass = 1;

const static int kPoseNumClass = 1;
const static int kNumberOfPoints = 17;  // number of keypoints total
// obb model's number of classes
constexpr static int kObbNumClass = 15;
const static int kObbNe = 1;  // number of extra parameters
const static int kBatchSize = 1;
const static int kGpuId = 0;

// 图片输入的宽高，与训练时的尺寸一致(for seg and det )
const static int kInputH = 640;
const static int kInputW = 640;

const static int kObbInputH = 1024;
const static int kObbInputW = 1024;
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;
const static float kConfThreshKeypoints = 0.5f;  // keypoints confidence
const static int kMaxInputImageSize = 3000 * 3000;
const static int kMaxNumOutputBbox = 1000;
//Quantization input image folder path
const static char* kInputQuantizationFolder = "./coco_calib";

// 分类模型类别数
constexpr static int kClsNumClass = 3;
// 分类模型图片输入的宽高，与训练时的尺寸一致
constexpr static int kClsInputH = 640;
constexpr static int kClsInputW = 640;
