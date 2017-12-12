//
// Created by Harpreet Singh on 10/30/17.
//

#include "ChangeContextBinary.h"

std::vector<cv::KeyPoint> ChangeDetector::ChangeContextBinary::getKeyPoints(cv::Mat &pImage, cv::Mat &pMask) {
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();

    std::vector<cv::KeyPoint> keyPoints;
    if (!pMask.empty()) {
        detector->detect(pImage, keyPoints, pMask);
    } else {
        detector->detect(pImage, keyPoints);
    }

    return keyPoints;
}

cv::Mat
ChangeDetector::ChangeContextBinary::getKeyPointDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pKeyPoints) {
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

    cv::Mat descriptors;
    extractor->compute(pImage, pKeyPoints, descriptors);

    return descriptors;
}

std::vector<std::vector<cv::DMatch>> ChangeDetector::ChangeContextBinary::matchFeaturesKNN(cv::Mat &pQueryDescriptors,
                                                                                           cv::Mat &pReferenceDescriptors,
                                                                                           int pK) {
    std::vector<std::vector<cv::DMatch>> kNearestMatches;
    cv::FlannBasedMatcher flannMatcher(new cv::flann::LshIndexParams(12, 12, 2));
    flannMatcher.knnMatch(pQueryDescriptors, pReferenceDescriptors, kNearestMatches, pK);

    return kNearestMatches;
}
