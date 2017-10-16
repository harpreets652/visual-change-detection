//
// Created by Harpreet Singh on 10/15/17.
//

#ifndef CHANGE_DETECTION_SURFDESCRIPTOR_H
#define CHANGE_DETECTION_SURFDESCRIPTOR_H

#include "Descriptor.h"
#include <opencv2/xfeatures2d.hpp>

class SURFDescriptor : public Descriptor {
public:
    SURFDescriptor() {}

    ~SURFDescriptor() {}

    cv::Mat getDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures);
};


#endif //CHANGE_DETECTION_SURFDESCRIPTOR_H
