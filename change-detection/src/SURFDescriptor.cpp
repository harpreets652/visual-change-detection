//
// Created by Harpreet Singh on 10/15/17.
//

#include "SURFDescriptor.h"

cv::Mat SURFDescriptor::getDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures) {
    cv::Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create(400);

    cv::Mat descriptors;
    extractor->compute(pImage, pFeatures, descriptors);

    return descriptors;
}