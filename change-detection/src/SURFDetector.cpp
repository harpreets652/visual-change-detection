//
// Created by Harpreet Singh on 10/15/17.
//

#include "SURFDetector.h"

std::vector<cv::KeyPoint> SURFDetector::getFeatures(cv::Mat &pImage) {
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

    std::vector<cv::KeyPoint> keyPoints;
    detector->detect(pImage, keyPoints);

    return keyPoints;
}