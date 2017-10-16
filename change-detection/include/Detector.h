//
// Created by Harpreet Singh on 10/15/17.
//

#ifndef CHANGE_DETECTION_DETECTOR_H
#define CHANGE_DETECTION_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class Detector {
public:
    Detector() {}

    virtual ~Detector() {}

    virtual std::vector<cv::KeyPoint> getFeatures(cv::Mat &pImage)=0;
};

#endif //CHANGE_DETECTION_DETECTOR_H
