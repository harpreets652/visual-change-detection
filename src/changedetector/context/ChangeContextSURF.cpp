//
// Created by Harpreet Singh on 10/28/17.
//

#include "ChangeContextSURF.h"

std::vector<cv::KeyPoint> ChangeDetector::ChangeContextSURF::getKeyPoints(cv::Mat &pImage, cv::Mat &pMask) {
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

    std::vector<cv::KeyPoint> keyPoints;
    if (!pMask.empty()) {
        detector->detect(pImage, keyPoints, pMask);
    } else {
        detector->detect(pImage, keyPoints);
    }

    return keyPoints;
}

cv::Mat
ChangeDetector::ChangeContextSURF::getKeyPointDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pKeyPoints) {
    cv::Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create(400);

    cv::Mat descriptors;
    extractor->compute(pImage, pKeyPoints, descriptors);

    return descriptors;
}

std::vector<std::vector<cv::DMatch>> ChangeDetector::ChangeContextSURF::matchFeaturesKNN(cv::Mat &pQueryDescriptors,
                                                                                         cv::Mat &pReferenceDescriptors,
                                                                                         int pK) {
    std::vector<std::vector<cv::DMatch>> kNearestMatches;
    cv::FlannBasedMatcher flannMatcher;
    flannMatcher.knnMatch(pQueryDescriptors, pReferenceDescriptors, kNearestMatches, pK);

    return kNearestMatches;
}