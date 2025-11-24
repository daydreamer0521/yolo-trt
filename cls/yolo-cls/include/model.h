/*
 * yolo11 模型构建函数声明
 * 这些函数会从给定的 .wts 权重文件读取权重并使用 TensorRT API 构建对应的网络，
 * 最终返回序列化的引擎（IHostMemory*），调用者负责将其写入磁盘。
 *
 * 参数说明（通用）:
 *  - builder: TensorRT 的 IBuilder 对象
 *  - config:  IBuilderConfig 配置对象（可用于设置 FP16/INT8 等选项）
 *  - dt:      数据类型（例如 DataType::kFLOAT）
 *  - wts_path: 权重文件路径（.wts）
 *  - gd, gw:  宽高/深度缩放系数（由模型类型 n/s/m/l/x 决定）
 *  - type:    模型子类型标识（例如 'n','s','m','l','x'）
 *  - max_channels: 网络中的最大通道数（受模型规模控制）
 */
#pragma once

#include <assert.h>
#include <string>
#include "NvInfer.h"

nvinfer1::IHostMemory* buildEngineYolo11Cls(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            std::string& type, int max_channels);

nvinfer1::IHostMemory* buildEngineYolo11Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type);

nvinfer1::IHostMemory* buildEngineYolo11Seg(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type);

nvinfer1::IHostMemory* buildEngineYolo11Pose(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                             int& max_channels, std::string& type);

nvinfer1::IHostMemory* buildEngineYolo11Obb(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type);
