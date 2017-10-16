//
// Created by Harpreet Singh on 10/15/17.
//

#ifndef CHANGE_DETECTION_SURFDETECTOR_H
#define CHANGE_DETECTION_SURFDETECTOR_H

#include "Detector.h"
#include <opencv2/xfeatures2d.hpp>

class SURFDetector : public Detector {
public:
    SURFDetector() {};

    ~SURFDetector() {};

    std::vector<cv::KeyPoint> getFeatures(cv::Mat &pImage);
};


#endif //CHANGE_DETECTION_SURFDETECTOR_H
