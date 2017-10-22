//
// Created by Harpreet Singh on 10/16/17.
//

#include "FlannMatcher.h"

/*
 * Copied implementation from
 * https://docs.opencv.org/3.0-beta/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
 */

std::vector<cv::KeyPoint> ChangeDetector::FlannMatcher::featureDetection(cv::Mat &pImage) {
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

    std::vector<cv::KeyPoint> keyPoints;
    detector->detect(pImage, keyPoints);

    return keyPoints;
}

cv::Mat ChangeDetector::FlannMatcher::featureDescription(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures) {
    cv::Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create(400);

    cv::Mat descriptors;
    extractor->compute(pImage, pFeatures, descriptors);

    return descriptors;
}

void ChangeDetector::FlannMatcher::preProcess() {
    std::cout << "nothing to pre-process" << std::endl;
}

void ChangeDetector::FlannMatcher::match(cv::Mat &pDescriptorOne, cv::Mat &pDescriptorTwo) {
    cv::FlannBasedMatcher flannMatcher;

    flannMatcher.match(pDescriptorOne, pDescriptorTwo, matchedFeatures);
}

void ChangeDetector::FlannMatcher::postProcess() {
    double maxDistance = 0;
    double minDistance = 100;
    std::vector<cv::DMatch> goodMatches;
    for (int i = 0; i < queryImage.keyPointDescriptors.rows; i++) {
        double distance = matchedFeatures[i].distance;
        if (distance < minDistance) {
            minDistance = distance;
        }

        if (distance > maxDistance) {
            maxDistance = distance;
        }
    }

    for (int i = 0; i < queryImage.keyPointDescriptors.rows; i++) {
        if (matchedFeatures[i].distance <= std::max(2 * minDistance, 0.1)) {
            goodMatches.push_back(matchedFeatures[i]);
        }
    }

    minDistanceOfMatches = minDistance;
    maxDistanceOfMatches = maxDistance;

    std::cout << "Maximum distance in matched features: " << maxDistance << std::endl;
    std::cout << "Minimum distance in matched features: " << minDistance << std::endl;

/* Note~ this isn't necessary
    cv::Mat imageOfMatches;
    cv::drawMatches(queryImage.imageData, queryImage.keyPoints,
                    trainingImage.imageData, trainingImage.keyPoints,
                    goodMatches, imageOfMatches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    cv::namedWindow("Image with matches", cv::WINDOW_KEEPRATIO);
    cv::imshow("Image with matches", imageOfMatches);
*/

    std::cout << "Good matches found:" << std::endl;
    for (const auto &feature : goodMatches) {
        std::cout << "KeyPoint 1: " << feature.queryIdx << ", KeyPoint 2: " << feature.trainIdx << std::endl;
    }

//    cv::waitKey(0);
}

std::vector<cv::DMatch> ChangeDetector::FlannMatcher::getMatchedFeatures(
        std::function<bool(cv::DMatch &, std::pair<double, double>)> &pCriteria) {

    std::vector<cv::DMatch> matchesWithinCriteria;
    std::pair<double, double> minMax = getMinAndMaxFeatureMatchDistances();

    for (int i = 0; i < queryImage.keyPointDescriptors.rows; i++) {
        if (pCriteria(matchedFeatures[i], minMax)) {
            matchesWithinCriteria.push_back(matchedFeatures[i]);
        }
    }

    return matchesWithinCriteria;
}

std::pair<double, double> ChangeDetector::FlannMatcher::getMinAndMaxFeatureMatchDistances() {
    return {minDistanceOfMatches, maxDistanceOfMatches};
};

