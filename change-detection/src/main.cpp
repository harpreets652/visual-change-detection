#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "SURFDetector.h"
#include "SURFDescriptor.h"
#include "FeatureMatch.h"
#include "FlannMatcher.h"

void showImage(std::string &imageName);

void showImage(cv::Mat &image);

void featureDetectorSURF(std::string &imageName);

void featureMatchUsingFlann(std::string &imageOne, std::string &imageTwo);

int main() {
    std::string imageName = "../resources/chessboard.png";
    std::string imageOne = "../resources/scene.png";
    std::string imageTwo = "../resources/crackerBox.png";

//    featureDetectorSURF(imageName);
    featureMatchUsingFlann(imageTwo, imageOne);

    return 0;
}

void featureMatchUsingFlann(std::string &imageOne, std::string &imageTwo) {
    ChangeDetector::FeatureMatch *featureMatcher = new ChangeDetector::FlannMatcher();

    ChangeDetector::Detector *surfDetector = new ChangeDetector::SURFDetector();
    ChangeDetector::Descriptor *surfDescriptor = new ChangeDetector::SURFDescriptor();

    featureMatcher->withDetector(surfDetector)
            ->withDescriptor(surfDescriptor)
            ->execute(imageOne, imageTwo);
}

void featureDetectorSURF(std::string &imageName) {
    cv::Mat inputImage = cv::imread(imageName, cv::IMREAD_GRAYSCALE);

    if (!inputImage.data) {
        std::cerr << "Unable to open input image " << imageName << std::endl;
        return;
    }

    ChangeDetector::Detector *surfDetector = new ChangeDetector::SURFDetector();
    std::vector<cv::KeyPoint> keyPoints = surfDetector->getFeatures(inputImage);;

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