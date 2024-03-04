
#include "log.h"
#include "preprocess.h"
#include "calibrator.h"
#include <unistd.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;

const int nHeight  {224};
const int nWidth   {224};
const int nChannel {3};
const std::string onnxFile  {"../data/resnet18.onnx"};
const std::string imgFile   {"../data/images/reflex_camera.jpeg"};
const std::string labelFile {"../data/class_labels.txt"};
const std::string cacheFile {"./int8.cache"};
const std::string calibrationDataDir {"../data/images"};
const int         nCalibration {1};
const int         calibrationBatchSize {2};

static Logger gLogger(ILogger::Severity::kERROR);


int main(int argc, char* argv[]) {
    bool bFP16Mode {false};
    bool bINT8Mode {false};
    for (int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fp16")
            bFP16Mode = true;
        if (arg == "--int8")
            bINT8Mode = true;
    }

    std::string trtFile = (bFP16Mode) ? "./resnet18-fp16.plan" :
                          (bINT8Mode) ? "./resnet18-int8.plan" :
                                        "./resnet18-fp32.plan";

    CHECK(cudaSetDevice(0));
    ICudaEngine* engine = nullptr;

    // 检查文件是否存在
    if (access(trtFile.c_str(), F_OK) == 0) {
        // 打开文件
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int fsize = 0;
        // 获取文件大小
        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        // 重置文件指针
        engineFile.seekg(0, engineFile.beg);
        // 读取文件内容
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        // 检查读到的文件是否为空
        if (engineString.size() == 0) {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime* runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) {
            std::cout << "Failed loading engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else {
        IBuilder*             builder     = createInferBuilder(gLogger);
        INetworkDefinition*   network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig*       config      = builder->createBuilderConfig();
        IInt8Calibrator*      pCalibrator = nullptr;
        if (bFP16Mode) {
            config->setFlag(BuilderFlag::kFP16);
        }
        
        if (bINT8Mode) {
            config->setFlag(BuilderFlag::kINT8);
            Dims32 inputShape {4, {calibrationBatchSize, nChannel, nHeight, nWidth}};
            pCalibrator = new MyCalibrator(calibrationDataDir, nCalibration, inputShape, cacheFile);
            if (pCalibrator == nullptr) {
                std::cout << std::string("Failed getting Calibrator for Int8!") << std::endl;
                return 1;
            }
            config->setInt8Calibrator(pCalibrator);
        } 

        // 加载 ONNX
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity))) {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                auto* error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }

            return 1;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        ITensor* inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, nChannel, nHeight, nWidth}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {3, nChannel, nHeight, nWidth}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, nChannel, nHeight, nWidth}});
        config->addOptimizationProfile(profile);

        // 序列化网络
        IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0) {
            std::cout << "Failed building serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // 推理引擎
        IRuntime* runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) {
            std::cout << "Failed building engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr) {
            delete pCalibrator;
        }
        
        // 保存序列化网络
        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile) {
            std::cout << "Failed opening file to write" << std::endl;
            return 1;
        }
        engineFile.write(static_cast<char*>(engineString->data()), engineString->size());
        if (engineFile.fail()) {
            std::cout << "Failed saving .plan file!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    int nIO = engine->getNbIOTensors();
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i) {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setInputShape(vTensorName[0].c_str(), Dims32 {4, {1, nChannel, nHeight, nWidth}});

    for (int i = 0; i < nIO; ++i) {
        std::cout << std::string(i == 0 ? "Input [" : "Output[");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }

    // 计算输入输出 Tensor 的大小
    std::vector<int> vTensorSize(nIO, 0);
    for (int i = 0; i < nIO; ++i) {
        Dims32 dim = context->getTensorShape(vTensorName[i].c_str());
        // size: Tensor 中元素个数
        int size = 1;
        for (int j = 0; j < dim.nbDims; ++j) {
            size *= dim.d[j];
        }
        // size * 每个元素的大小
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    void* inputHost = (void*)new char[vTensorSize[0]];
    void* outputHost = (void*)new char[vTensorSize[1]];
    void* inputDevice;
    void* outputDevice;
    CHECK(cudaMalloc(&inputDevice, vTensorSize[0]));
    CHECK(cudaMalloc(&outputDevice, vTensorSize[1]));

    std::vector<float> img = loadImg(imgFile, nWidth, nHeight, nChannel);

    memcpy(inputHost, img.data(), vTensorSize[0]);

    // H2D
    CHECK(cudaMemcpy(inputDevice, inputHost, vTensorSize[0], cudaMemcpyHostToDevice));
    context->setTensorAddress(vTensorName[0].c_str(), inputDevice);
    context->setTensorAddress(vTensorName[1].c_str(), outputDevice);
    // 推理
    context->enqueueV3(0);
    // D2H
    CHECK(cudaMemcpy(outputHost, outputDevice, vTensorSize[1], cudaMemcpyDeviceToHost));

    // printArrayInformation((float *)inputHost, context->getTensorShape(vTensorName[0].c_str()), vTensorName[0], true, true);
    // printArrayInformation((float *)outputHost, context->getTensorShape(vTensorName[1].c_str()), vTensorName[1], true, true);
    
    float* outputData = reinterpret_cast<float*>(outputHost);
    float maxValue = outputData[0];
    int maxIdx = 0;
    for (int i = 1; i < 1000; ++i) {
        if (outputData[i] > maxValue) {
            maxValue = outputData[i];
            maxIdx = i;
        }
    }

    std::ifstream file(labelFile);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file." << std::endl;
        return -1;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    std::cout << "Prediction: " << lines[maxIdx] << std::endl;
    std::cout << "Confidence: " << maxValue << std::endl;

    delete[] (char*)inputHost;
    delete[] (char*)outputHost;
    CHECK(cudaFree(inputDevice));
    CHECK(cudaFree(outputDevice));
    return 0;
}