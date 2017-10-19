//
// Created by Harpreet Singh on 10/16/17.
//

#include "BriefDescriptor.h"

cv::Mat ChangeDetector::BriefDescriptor::getDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures) {
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

    cv::Mat descriptors;
    extractor->compute(pImage, pFeatures, descriptors);

    return descriptors;
}