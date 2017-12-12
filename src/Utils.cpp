//
// Created by Harpreet Singh on 11/1/17.
//

#include "Utils.h"

void showImage(cv::Mat &image, std::string pTitle) {
    if (image.empty()) {
        std::cerr << "Image may not be empty" << std::endl;
        return;
    }

    cv::namedWindow(pTitle, cv::WINDOW_KEEPRATIO);
    cv::imshow(pTitle, image);

    cv::waitKey(0);
    cv::destroyWindow(pTitle);
}

void showImage(std::string &imageName, std::string &pTitle) {
    cv::Mat image = cv::imread(imageName);

    if (image.empty()) {
        std::cerr << "Unable to read image " << imageName << std::endl;
    }

    showImage(image, pTitle);
}

cv::Mat getSaliency(cv::Mat &pImage) {
    cv::Mat saliencyImage;

    cv::Ptr<cv::saliency::Saliency> saliencyAlgorithm = cv::saliency::StaticSaliencySpectralResidual::create();
    saliencyAlgorithm->computeSaliency(pImage, saliencyImage);

    return saliencyImage;
}