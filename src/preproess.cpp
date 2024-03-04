#include "preprocess.h"

std::vector<float> loadImg(const std::string filename, int width, int height, int channel) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Unable to read image file." << std::endl;
        return std::vector<float>();
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    cv::Scalar meanData(0.485, 0.456, 0.406);
    cv::Scalar stdData(0.229, 0.224, 0.225);
    cv::subtract(image, meanData, image);
    cv::divide(image, stdData, image);

    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    std::vector<float> data(channel * height * width);
    int idx = 0;
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                data[idx++] = channels[c].at<float>(h, w);
            }
        }
    }
    return data;
}