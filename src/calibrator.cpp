#include "calibrator.h"

using namespace nvinfer1;

namespace fs = std::filesystem;

MyCalibrator::MyCalibrator(const std::string &calibrationDataDir, const int nCalibration, const Dims32 dim, const std::string &cacheFile):
    nCalibration(nCalibration), dim(dim), cacheFile(cacheFile), iBatch(0) {

    
    for (const auto& entry : fs::directory_iterator(calibrationDataDir)) {
        if (fs::is_regular_file(entry)) {
            files.push_back(entry.path().string());
        }
    }

    nBatch = files.size() / dim.d[0];
    nElement = 1;
    for (int i = 0; i < dim.nbDims; ++i) {
        nElement *= dim.d[i];
    }
    bufferSize = sizeof(float) * nElement;
    cudaMalloc((void **)&bufferD, bufferSize);

    return;
}

MyCalibrator::~MyCalibrator() noexcept {
    if (bufferD != nullptr) {
        cudaFree(bufferD);
    }
    return;
}

int32_t MyCalibrator::getBatchSize() const noexcept {
    return dim.d[0];
}

bool MyCalibrator::getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept {
    if (iBatch < nBatch) {
        for (int i = 0; i < dim.d[0]; ++i) {
            std::vector<float> img = loadImg(files[iBatch*dim.d[0]+i], dim.d[3], dim.d[2], dim.d[1]);
            cudaMemcpy(&bufferD[i*img.size()], img.data(), img.size()*sizeof(float), cudaMemcpyHostToDevice);
        }
        bindings[0] = bufferD;
        iBatch++;
        return true;
    }
    else {
        return false;
    }
}

void const *MyCalibrator::readCalibrationCache(std::size_t &length) noexcept {
    std::fstream f;
    f.open(cacheFile, std::fstream::in);
    if (f.fail()) {
        std::cout << "Failed finding cache file!" << std::endl;
        return nullptr;
    }
    char *ptr = new char[length];
    if (f.is_open()) {
        f >> ptr;
    }
    return ptr;
}

void MyCalibrator::writeCalibrationCache(void const* ptr, std::size_t length) noexcept {
    std::ofstream f(cacheFile, std::ios::binary);
    if (f.fail()) {
        std::cout << "Failed opening cache file to write!" << std::endl;
        return;
    }
    f.write(static_cast<char const *>(ptr), length);
    if (f.fail()) {
        std::cout << "Failed saving cache file!" << std::endl;
        return;
    }
    f.close();
}
