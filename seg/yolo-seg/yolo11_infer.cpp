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
// output buffer sizes (match yolo11_seg.cpp)
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);

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

void prepare_buffer(ICudaEngine* engine, float** input_dev, float** output_dev, float** output_seg_dev,
                    float** output_host, float** output_seg_host, float** decode_ptr_host, float** decode_ptr_dev,
                    const std::string& cuda_post_process) {
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    const int outputSegIndex = engine->getBindingIndex(kProtoTensorName);
    if (inputIndex != 0 || outputIndex != 1 || outputSegIndex != 2) {
        std::cerr << "Unexpected binding indices" << std::endl;
    }
    CUDA_CHECK(cudaMalloc((void**)input_dev, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_dev, kBatchSize * kOutputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_seg_dev, kBatchSize * kOutputSegSize * sizeof(float)));

    if (cuda_post_process == "c") {
        *output_host = new float[kBatchSize * kOutputSize];
        *output_seg_host = new float[kBatchSize * kOutputSegSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            throw std::runtime_error("GPU post process not supported for multiple batches");
        }
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        *output_seg_host = new float[kBatchSize * kOutputSegSize];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_dev, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) {
    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++) {
        cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
        auto rleft = dets[i].bbox[0];
        auto rtop = dets[i].bbox[1];
        auto rright = dets[i].bbox[0] + dets[i].bbox[2];
        auto rbottom = dets[i].bbox[1] + dets[i].bbox[3];
        int lx = std::max(0, (int)(rleft / 4));
        int ty = std::max(0, (int)(rtop / 4));
        int rx = std::min(kInputW/4, (int)(rright / 4));
        int by = std::min(kInputH/4, (int)(rbottom / 4));
        for (int x = lx; x < rx; x++) {
            for (int y = ty; y < by; y++) {
                float e = 0.0f;
                for (int j = 0; j < 32; j++) {
                    e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask_mat.at<float>(y, x) = e;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
        masks.push_back(mask_mat);
    }
    return masks;
}

void do_inference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output_host, float* output_seg_host,
                  int batchsize, float* decode_ptr_host, float* decode_ptr_dev, int model_bboxes, const std::string& cuda_post_process) {
    auto t0 = std::chrono::high_resolution_clock::now();
    context.enqueueV2(buffers, stream, nullptr);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output_host, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_seg_host, buffers[2], batchsize * kOutputSegSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_dev, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_dev, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_dev, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_dev, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_seg_host, buffers[2], batchsize * kOutputSegSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
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
    bool save_result = true;

    if (argc >= 5) {
        std::string arg4 = argv[4];
        if (arg4 == "c" || arg4 == "g") {
            cuda_post = arg4;
            if (argc >= 6) save_result = std::stoi(argv[5]);
        } else {
            // Assume arg4 is save_result, and cuda_post defaults to "c"
            try {
                save_result = std::stoi(arg4);
            } catch (...) {
                std::cerr << "Invalid argument for postprocess or save_result: " << arg4 << std::endl;
                return -1;
            }
        }
    }

    std::cout << "Postprocess mode: " << cuda_post << ", Save result: " << save_result << std::endl;

    // read labels
    std::unordered_map<int,std::string> labels_map;
    read_labels(labels_file, labels_map);
    std::cout << "Loaded " << labels_map.size() << " labels." << std::endl;

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

    auto out_dims = engine->getBindingDimensions(1);
    int model_bboxes = out_dims.d[0];

    float* input_dev=nullptr; float* output_dev=nullptr; float* output_seg_dev=nullptr;
    float* output_host=nullptr; float* output_seg_host=nullptr; float* decode_ptr_host=nullptr; float* decode_ptr_dev=nullptr;
    prepare_buffer(engine, &input_dev, &output_dev, &output_seg_dev, &output_host, &output_seg_host, &decode_ptr_host, &decode_ptr_dev, cuda_post);

    void* buffers[3];
    buffers[0] = input_dev; buffers[1] = output_dev; buffers[2] = output_seg_dev;

    if (is_video_file(input_path)) {
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) { std::cerr << "open video failed" << std::endl; return -1; }
        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        
        cv::VideoWriter writer;
        if (save_result) {
            writer.open("out_video.mp4", cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(w,h));
        }

        cv::Mat frame;
        while (cap.read(frame)) {
            std::vector<cv::Mat> batch; batch.push_back(frame);
            cuda_batch_preprocess(batch, input_dev, kInputW, kInputH, stream);
            do_inference(*context, stream, buffers, output_host, output_seg_host, kBatchSize, decode_ptr_host, decode_ptr_dev, model_bboxes, cuda_post);
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post == "c") {
                batch_nms(res_batch, output_host, batch.size(), kOutputSize, kConfThresh, kNmsThresh);
                for (size_t b=0;b<batch.size();b++){
                    auto& res = res_batch[b];
                    std::cout << "Batch " << b << " has " << res.size() << " detections." << std::endl;
                    auto masks = process_mask(&output_seg_host[b * kOutputSegSize], kOutputSegSize, res);
                    draw_mask_bbox(batch[b], res, masks, labels_map);
                }
            } else if (cuda_post == "g") {
                batch_process(res_batch, decode_ptr_host, batch.size(), bbox_element, batch);
                for (size_t b=0;b<batch.size();b++){
                    auto& res = res_batch[b];
                    std::cout << "Batch " << b << " has " << res.size() << " detections (GPU)." << std::endl;
                    auto masks = process_mask(&output_seg_host[b * kOutputSegSize], kOutputSegSize, res);
                    draw_mask_bbox(batch[b], res, masks, labels_map);
                }
            } else {
                std::cerr << "GPU postprocess for seg not supported in this tool" << std::endl;
            }
            
            if (save_result && writer.isOpened()) {
                writer.write(batch[0]);
            }
            cv::imshow("result", batch[0]);
            if (cv::waitKey(1) == 27) break;
        }
        if (save_result) writer.release();
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
            cuda_batch_preprocess(batch, input_dev, kInputW, kInputH, stream);
            do_inference(*context, stream, buffers, output_host, output_seg_host, kBatchSize, decode_ptr_host, decode_ptr_dev, model_bboxes, cuda_post);
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post == "c") {
                batch_nms(res_batch, output_host, batch.size(), kOutputSize, kConfThresh, kNmsThresh);
                for (size_t b=0;b<batch.size();b++){
                    auto& res = res_batch[b];
                    auto masks = process_mask(&output_seg_host[b * kOutputSegSize], kOutputSegSize, res);
                    draw_mask_bbox(batch[b], res, masks, labels_map);
                    
                    if (save_result) {
                        std::string filename = names[b];
                        size_t last_slash = filename.find_last_of('/');
                        if (last_slash != std::string::npos) {
                            filename = filename.substr(last_slash + 1);
                        }
                        std::string outname = "result_" + filename;
                        cv::imwrite(outname, batch[b]);
                        std::cout << "Wrote " << outname << std::endl;
                    }
                }
            } else if (cuda_post == "g") {
                batch_process(res_batch, decode_ptr_host, batch.size(), bbox_element, batch);
                for (size_t b=0;b<batch.size();b++){
                    auto& res = res_batch[b];
                    auto masks = process_mask(&output_seg_host[b * kOutputSegSize], kOutputSegSize, res);
                    draw_mask_bbox(batch[b], res, masks, labels_map);
                    
                    if (save_result) {
                        std::string filename = names[b];
                        size_t last_slash = filename.find_last_of('/');
                        if (last_slash != std::string::npos) {
                            filename = filename.substr(last_slash + 1);
                        }
                        std::string outname = "result_" + filename;
                        cv::imwrite(outname, batch[b]);
                        std::cout << "Wrote " << outname << std::endl;
                    }
                }
            } else {
                std::cerr << "GPU postprocess for seg not supported in this tool" << std::endl;
            }
        }
    }

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(input_dev));
    CUDA_CHECK(cudaFree(output_dev));
    CUDA_CHECK(cudaFree(output_seg_dev));
    CUDA_CHECK(cudaFree(decode_ptr_dev));
    delete[] decode_ptr_host; delete[] output_host; delete[] output_seg_host;
    cuda_preprocess_destroy();
    delete context; delete engine; delete runtime;
    return 0;
}
