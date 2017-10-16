//
// Created by Harpreet Singh on 10/15/17.
//

#ifndef CHANGE_DETECTION_DESCRIPTOR_H
#define CHANGE_DETECTION_DESCRIPTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace ChangeDetector {
    class Descriptor {
    public:
        Descriptor() {}

        virtual ~Descriptor() {}

        virtual cv::Mat getDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures)=0;
    };
}

#endif //CHANGE_DETECTION_DESCRIPTOR_H
