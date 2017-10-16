#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

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

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(inputImage, keypoints);

    cv::Mat keyPointsImage;

    cv::drawKeypoints(inputImage, keypoints, keyPointsImage);

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