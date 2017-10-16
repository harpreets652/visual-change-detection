#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "SURFDetector.h"
#include "SURFDescriptor.h"

void showImage(std::string &imageName);

void showImage(cv::Mat &image);

void featureDetectorSURF(std::string &imageName);

int main() {
    std::string imageName = "../resources/chessboard.png";
    featureDetectorSURF(imageName);

    return 0;
}

void featureDetectorSURF(std::string &imageName) {
    cv::Mat inputImage = cv::imread(imageName, cv::IMREAD_GRAYSCALE);

    if (!inputImage.data) {
        std::cerr << "Unable to open input image " << imageName << std::endl;
        return;
    }

    Detector *surfDetector = new SURFDetector();
    std::vector<cv::KeyPoint> keyPoints = surfDetector->getFeatures(inputImage);;

    Descriptor *surfDescriptor = new SURFDescriptor();
    cv::Mat descriptors = surfDescriptor->getDescriptors(inputImage, keyPoints);

    cv::Mat keyPointsImage;
    cv::drawKeypoints(inputImage, keyPoints, keyPointsImage);

    showImage(keyPointsImage);
}

void showImage(std::string &imageName) {
    cv::Mat testing = cv::imread(imageName);

    if (!testing.data) {
        std::cerr << "unable to read image" << std::endl;
    }

    showImage(testing);
}

void showImage(cv::Mat &image) {
    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::imshow("image name", image);

    cv::waitKey(0);
}