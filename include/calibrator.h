#pragma once

#include "preprocess.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class MyCalibrator : public IInt8EntropyCalibrator2
{
private:
    int         nCalibration {0};
    int         nElement {0};
    size_t      bufferSize {0};
    int         nBatch {0};
    int         iBatch {0};
    float*      bufferD {nullptr};
    Dims32      dim;
    std::string cacheFile {""};

    std::vector<std::string> files;

public:
    MyCalibrator(const std::string &calibrationDataDir, const int nCalibration, const Dims32 inputShape, const std::string &cacheFile);
    ~MyCalibrator() noexcept;
    int32_t     getBatchSize() const noexcept;
    bool        getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept;
    void const* readCalibrationCache(std::size_t &length) noexcept;
    void        writeCalibrationCache(void const* ptr, std::size_t length) noexcept;
};