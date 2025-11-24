#include <fstream>
#include <iostream>
#include "logging.h"
#include "model.h"

using namespace nvinfer1;

Logger gLogger;

int main(int argc, char** argv) {
    // 改改路径
    const std::string wts_name = "/home/rc/yolo11trt/models/best.wts";    // 在这里设置你的 .wts 文件路径
    const std::string engine_name = "/home/rc/yolo11trt/models/best.engine"; // 在这里设置输出 engine 文件路径
    const char model_type = 'n';                           // 模型后缀
    // 支持安全模式：命令行参数 --dry-run 或 环境变量 SAFE_MODE=1
    bool dry_run = false;
    if (argc > 1 && std::string(argv[1]) == "--dry-run") {
        dry_run = true;
    }
    const char* safe_env = std::getenv("SAFE_MODE");
    if (safe_env && std::string(safe_env) == "1") dry_run = true;
    // 将 model_type 映射到 (gd, gw, max_channels)
    float gd = 0.0f, gw = 0.0f;
    int max_channels = 0;
    switch (model_type) {
        case 'n': gd = 0.50f; gw = 0.25f; max_channels = 1024; break;
        case 's': gd = 0.50f; gw = 0.50f; max_channels = 1024; break;
        case 'm': gd = 0.50f; gw = 1.00f; max_channels = 512; break;
        case 'l': gd = 1.00f; gw = 1.00f; max_channels = 512; break;
        case 'x': gd = 1.00f; gw = 1.50f; max_channels = 512; break;
        default:
            std::cerr << "脑子呢？？？自己输入的模型后缀不清楚吗" << std::endl;
            return -1;
    }
    
    if (dry_run) {
        std::cout << "[DRY-RUN] SAFE MODE enabled — will NOT build engine or call CUDA/TensorRT." << std::endl;
        std::ifstream in(wts_name, std::ios::binary | std::ios::ate);
        if (!in) {
            std::cerr << "[DRY-RUN] wts file not found: " << wts_name << std::endl;
            return -1;
        }
        auto size = in.tellg();
        std::cout << "[DRY-RUN] Found wts file: " << wts_name << " (" << size << " bytes)." << std::endl;
        std::cout << "[DRY-RUN] Would create engine file: " << engine_name << " with model type '" << model_type
                  << "' and params gd=" << gd << " gw=" << gw << " max_channels=" << max_channels << std::endl;
        return 0;
    }

    // 创建 TensorRT builder 和 config
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // 使用已实现的构建函数生成序列化引擎
    std::string type = std::string(1, model_type);
    IHostMemory* serialized_engine = buildEngineYolo11Seg(builder, config, DataType::kFLOAT, wts_name, gd, gw,
                                                          max_channels, type);
    if (!serialized_engine) {
        std::cerr << "构建引擎失败:( buildEngineYolo11Seg 返回空" << std::endl;
        delete config;
        delete builder;
        return -1;
    }

    // 将序列化的引擎写入到文件
    std::ofstream out(engine_name, std::ios::binary);
    if (!out) {
        std::cerr << "再看看你的输出路径呢？！" << std::endl;
        delete serialized_engine;
        delete config;
        delete builder;
        return -1;
    }
    out.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    out.close();

    delete serialized_engine;
    delete config;
    delete builder;

    std::cout << "已生成 engine 文件>< :" << engine_name << std::endl;
    return 0;
}
