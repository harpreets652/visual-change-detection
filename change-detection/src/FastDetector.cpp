//
// Created by Harpreet Singh on 10/16/17.
//

#include "FastDetector.h"

std::vector<cv::KeyPoint> ChangeDetector::FastDetector::getFeatures(cv::Mat &pImage) {
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();

    std::vector<cv::KeyPoint> keyPoints;
    detector->detect(pImage, keyPoints);

    return keyPoints;
}
