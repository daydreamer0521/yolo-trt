#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "logging.h"
#include "config.h"
#include "cuda_utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"

using namespace nvinfer1;

Logger gLogger;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

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

void prepare_buffer(ICudaEngine* engine, float** input_dev, float** output_dev,
                    float** output_host, float** decode_ptr_host, float** decode_ptr_dev,
                    const std::string& cuda_post_process) {
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    if (inputIndex != 0 || outputIndex != 1) {
        std::cerr << "Unexpected binding indices" << std::endl;
    }
    CUDA_CHECK(cudaMalloc((void**)input_dev, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_dev, kBatchSize * kOutputSize * sizeof(float)));

    if (cuda_post_process == "c") {
        *output_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            throw std::runtime_error("GPU post process not supported for multiple batches");
        }
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_dev, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void do_inference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output_host,
                  int batchsize, float* decode_ptr_host, float* decode_ptr_dev, int model_bboxes, const std::string& cuda_post_process) {
    auto t0 = std::chrono::high_resolution_clock::now();
    context.enqueueV2(buffers, stream, nullptr);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output_host, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_dev, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_dev, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_dev, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_dev, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
}

// Helper to draw bbox with labels from map
void draw_bbox_with_labels(cv::Mat& img, std::vector<Detection>& dets, std::unordered_map<int, std::string>& labels_map) {
    static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17,
                                           0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF,
                                           0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};
    for (size_t i = 0; i < dets.size(); i++) {
        auto color = colors[(int)dets[i].class_id % colors.size()];
        auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);
        cv::Rect r = get_rect(img, dets[i].bbox);
        cv::rectangle(img, r, bgr, 2);
        
        std::string label_name = labels_map.count((int)dets[i].class_id) ? labels_map[(int)dets[i].class_id] : std::to_string((int)dets[i].class_id);
        std::string text = label_name + " " + to_string_with_precision(dets[i].conf);
        
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        cv::Point topLeft(r.x, r.y - textSize.height);
        cv::Point bottomRight(r.x + textSize.width, r.y + textSize.height);
        cv::rectangle(img, topLeft, bottomRight, bgr, -1);
        cv::putText(img, text, cv::Point(r.x, r.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);
    }
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
        std::cerr << "Usage:\n  " << argv[0] << " <engine> <input(image|video|dir)> <labels.txt> [c|g postprocess] [save_result(0|1)]\n";
        return -1;
    }
    std::string engine_name = argv[1];
    std::string input_path = argv[2];
    std::string labels_file = argv[3];
    std::string cuda_post = "c";
    if (argc >= 5) cuda_post = argv[4];
    bool save_result = true;
    if (argc >= 6) save_result = std::stoi(argv[5]);

    std::unordered_map<int,std::string> labels_map;
    read_labels(labels_file, labels_map);

    if (labels_map.empty()) {
        std::cerr << "Error: Labels file " << labels_file << " is empty or cannot be opened." << std::endl;
        return -1;
    }

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

    auto out_dims = engine->getBindingDimensions(1);
    int model_bboxes = out_dims.d[0];

    float* input_dev=nullptr; float* output_dev=nullptr;
    float* output_host=nullptr; float* decode_ptr_host=nullptr; float* decode_ptr_dev=nullptr;
    prepare_buffer(engine, &input_dev, &output_dev, &output_host, &decode_ptr_host, &decode_ptr_dev, cuda_post);

    void* buffers[2];
    buffers[0] = input_dev; buffers[1] = output_dev;

    if (is_video_file(input_path)) {
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) { std::cerr << "open video failed" << std::endl; return -1; }
        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter writer("out_video.mp4", cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(w,h));
        cv::Mat frame;
        while (cap.read(frame)) {
            std::vector<cv::Mat> batch; batch.push_back(frame);
            cuda_batch_preprocess(batch, input_dev, kInputW, kInputH, stream);
            do_inference(*context, stream, buffers, output_host, kBatchSize, decode_ptr_host, decode_ptr_dev, model_bboxes, cuda_post);
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post == "c") {
                batch_nms(res_batch, output_host, batch.size(), kOutputSize, kConfThresh, kNmsThresh);
                for (size_t b=0;b<batch.size();b++){
                    draw_bbox_with_labels(batch[b], res_batch[b], labels_map);
                }
            } else {
                batch_process(res_batch, decode_ptr_host, batch.size(), bbox_element, batch);
                for (size_t b=0;b<batch.size();b++){
                    draw_bbox_with_labels(batch[b], res_batch[b], labels_map);
                }
            }
            writer.write(batch[0]);
            cv::imshow("result", batch[0]);
            if (cv::waitKey(1) == 27) break;
        }
        writer.release(); cap.release();
    } else {
        std::vector<std::string> files;
        struct stat s;
        if (stat(input_path.c_str(), &s)==0 && S_ISDIR(s.st_mode)) {
            read_files_in_dir(input_path.c_str(), files);
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
            cuda_batch_preprocess(batch, input_dev, kInputW, kInputH, stream);
            do_inference(*context, stream, buffers, output_host, kBatchSize, decode_ptr_host, decode_ptr_dev, model_bboxes, cuda_post);
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post == "c") {
                batch_nms(res_batch, output_host, batch.size(), kOutputSize, kConfThresh, kNmsThresh);
                for (size_t b=0;b<batch.size();b++){
                    draw_bbox_with_labels(batch[b], res_batch[b], labels_map);
                    if (save_result) {
                        std::string outname = "_" + names[b].substr(names[b].find_last_of('/')+1);
                        cv::imwrite(outname, batch[b]);
                        std::cout << "Wrote " << outname << std::endl;
                    }
                }
            } else {
                batch_process(res_batch, decode_ptr_host, batch.size(), bbox_element, batch);
                for (size_t b=0;b<batch.size();b++){
                    draw_bbox_with_labels(batch[b], res_batch[b], labels_map);
                    if (save_result) {
                        std::string outname = "_" + names[b].substr(names[b].find_last_of('/')+1);
                        cv::imwrite(outname, batch[b]);
                        std::cout << "Wrote " << outname << std::endl;
                    }
                }
            }
        }
    }

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(input_dev));
    CUDA_CHECK(cudaFree(output_dev));
    if (cuda_post == "g") CUDA_CHECK(cudaFree(decode_ptr_dev));
    delete[] decode_ptr_host; delete[] output_host;
    cuda_preprocess_destroy();
    delete context; delete engine; delete runtime;
    return 0;
}
