//
// Created by Harpreet Singh on 10/16/17.
//

#ifndef CHANGE_DETECTION_BRIEFDESCRIPTOR_H
#define CHANGE_DETECTION_BRIEFDESCRIPTOR_H

#include "Descriptor.h"
#include <opencv2/xfeatures2d.hpp>

namespace ChangeDetector {
    class BriefDescriptor : public Descriptor {
        BriefDescriptor() {}

        ~BriefDescriptor() {}

        cv::Mat getDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures);
    };
}

#endif //CHANGE_DETECTION_BRIEFDESCRIPTOR_H
