//
// Created by Harpreet Singh on 10/16/17.
//

#ifndef CHANGE_DETECTION_FASTDETECTOR_H
#define CHANGE_DETECTION_FASTDETECTOR_H

#include "Detector.h"
#include <opencv2/xfeatures2d.hpp>

namespace ChangeDetector {
    class FastDetector : public Detector {
    public:
        FastDetector() {};

        ~FastDetector() {};

        std::vector<cv::KeyPoint> getFeatures(cv::Mat &pImage);
    };
}


#endif //CHANGE_DETECTION_FASTDETECTOR_H
