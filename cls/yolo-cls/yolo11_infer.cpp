#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <numeric>

#include "NvInfer.h"
#include "logging.h"
#include "config.h"
#include "cuda_utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"

using namespace nvinfer1;

Logger gLogger;

void deserialize_engine(const std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("read " + engine_name + " error!");
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    *runtime = createInferRuntime(gLogger);
    if (!*runtime) throw std::runtime_error("createInferRuntime failed");
    *engine = (*runtime)->deserializeCudaEngine(buf.data(), size);
    if (!*engine) throw std::runtime_error("deserializeCudaEngine failed");
    *context = (*engine)->createExecutionContext();
    if (!*context) throw std::runtime_error("createExecutionContext failed");
}

void prepare_buffer(ICudaEngine* engine, float** input_dev, float** output_dev, float** output_host, int& num_classes) {
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    if (inputIndex == -1 || outputIndex == -1) {
        std::cerr << "Invalid binding indices" << std::endl;
        exit(-1);
    }
    
    nvinfer1::Dims outDims = engine->getBindingDimensions(outputIndex);
    // Assuming output is [batch, num_classes]
    int outputVol = 1;
    for(int i=0; i<outDims.nbDims; ++i) outputVol *= outDims.d[i];
    num_classes = outputVol / kBatchSize;
    
    std::cout << "Model has " << num_classes << " classes." << std::endl;

    CUDA_CHECK(cudaMalloc((void**)input_dev, kBatchSize * 3 * kClsInputH * kClsInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_dev, kBatchSize * num_classes * sizeof(float)));

    *output_host = new float[kBatchSize * num_classes];
}

void do_inference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output_host, int batchsize, int num_classes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output_host, buffers[1], batchsize * num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
}

std::vector<float> softmax(const float* input, int size) {
    std::vector<float> output(size);
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
    return output;
}

std::vector<std::pair<int, float>> topk(const std::vector<float>& probs, int k) {
    int size = probs.size();
    if (k > size) k = size;
    std::vector<std::pair<int, float>> result;
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&probs](int i1, int i2) { return probs[i1] > probs[i2]; });
    for (int i = 0; i < k; ++i) {
        result.push_back({indices[i], probs[indices[i]]});
    }
    return result;
}

bool is_video_file(const std::string& path) {
    std::string ext;
    auto p = path.find_last_of('.');
    if (p == std::string::npos) return false;
    ext = path.substr(p+1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext=="mp4"||ext=="avi"||ext=="mov"||ext=="mkv");
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n  " << argv[0] << " <engine> <input(image|video|dir)> <labels.txt>\n";
        return -1;
    }
    std::string engine_name = argv[1];
    std::string input_path = argv[2];
    std::string labels_file = argv[3];

    // read labels
    std::unordered_map<int,std::string> labels_map;
    read_labels(labels_file, labels_map);

    // deserialize
    IRuntime* runtime=nullptr; ICudaEngine* engine=nullptr; IExecutionContext* context=nullptr;
    try {
        deserialize_engine(engine_name, &runtime, &engine, &context);
    } catch (const std::exception& e) {
        std::cerr << "Failed to deserialize engine: " << e.what() << std::endl;
        return -1;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);

    float* input_dev=nullptr; float* output_dev=nullptr;
    float* output_host=nullptr;
    int num_classes = 0;
    prepare_buffer(engine, &input_dev, &output_dev, &output_host, num_classes);

    void* buffers[2];
    buffers[0] = input_dev; buffers[1] = output_dev;

    if (is_video_file(input_path)) {
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) { std::cerr << "open video failed" << std::endl; return -1; }
        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        // cv::VideoWriter writer("out_video.mp4", cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(w,h));
        cv::Mat frame;
        while (cap.read(frame)) {
            std::vector<cv::Mat> batch; batch.push_back(frame);
            cuda_batch_preprocess(batch, input_dev, kClsInputW, kClsInputH, stream);
            do_inference(*context, stream, buffers, output_host, kBatchSize, num_classes);
            
            for (size_t b=0; b<batch.size(); b++) {
                std::vector<float> probs = softmax(output_host + b * num_classes, num_classes);
                auto top5 = topk(probs, 5);
                
                std::cout << "Frame result:" << std::endl;
                for (const auto& p : top5) {
                    std::string label = labels_map.count(p.first) ? labels_map[p.first] : std::to_string(p.first);
                    std::cout << "  " << label << ": " << p.second << std::endl;
                    
                    // Draw on image
                    std::string text = label + ": " + to_string_with_precision(p.second);
                    cv::putText(batch[b], text, cv::Point(10, 30 + (&p - &top5[0]) * 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                }
            }
            
            cv::imshow("result", batch[0]);
            if (cv::waitKey(1) == 27) break;
        }
        cap.release();
    } else {
        std::vector<std::string> files;
        struct stat s;
        if (stat(input_path.c_str(), &s)==0 && S_ISDIR(s.st_mode)) {
            DIR* dir = opendir(input_path.c_str());
            struct dirent* ent;
            while ((ent = readdir(dir))!=NULL) {
                std::string name = ent->d_name;
                if (name=="."||name=="..") continue;
                files.push_back(input_path + "/" + name);
            }
            closedir(dir);
            std::sort(files.begin(), files.end());
        } else {
            files.push_back(input_path);
        }
        for (size_t i=0;i<files.size(); i+=kBatchSize) {
            std::vector<cv::Mat> batch; std::vector<std::string> names;
            for (size_t j=i;j<i+kBatchSize && j<files.size(); ++j) {
                cv::Mat img = cv::imread(files[j]);
                if (img.empty()) continue;
                batch.push_back(img);
                names.push_back(files[j]);
            }
            if (batch.empty()) continue;
            cuda_batch_preprocess(batch, input_dev, kClsInputW, kClsInputH, stream);
            do_inference(*context, stream, buffers, output_host, kBatchSize, num_classes);
            
            for (size_t b=0; b<batch.size(); b++) {
                std::vector<float> probs = softmax(output_host + b * num_classes, num_classes);
                auto top5 = topk(probs, 5);
                
                std::cout << "Image: " << names[b] << std::endl;
                for (const auto& p : top5) {
                    std::string label = labels_map.count(p.first) ? labels_map[p.first] : std::to_string(p.first);
                    std::cout << "  " << label << ": " << p.second << std::endl;
                    
                    // Draw on image
                    std::string text = label + ": " + to_string_with_precision(p.second);
                    cv::putText(batch[b], text, cv::Point(10, 30 + (&p - &top5[0]) * 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                }
                
                std::string outname = "_" + names[b].substr(names[b].find_last_of('/')+1);
                cv::imwrite(outname, batch[b]);
                std::cout << "Wrote " << outname << std::endl;
            }
        }
    }

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(input_dev));
    CUDA_CHECK(cudaFree(output_dev));
    delete[] output_host;
    cuda_preprocess_destroy();
    delete context; delete engine; delete runtime;
    return 0;
}
