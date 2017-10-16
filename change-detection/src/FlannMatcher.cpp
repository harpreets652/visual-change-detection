//
// Created by Harpreet Singh on 10/16/17.
//

#include "FlannMatcher.h"

/*
 * Copied implementation from
 * https://docs.opencv.org/3.0-beta/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
 */
void ChangeDetector::FlannMatcher::preProcess() {
    std::cout << "nothing to pre-process" << std::endl;
}

void ChangeDetector::FlannMatcher::postProcess() {
    double maxDistance = 0;
    double minDistance = 100;
    std::vector<cv::DMatch> goodMatches;
    for (int i = 0; i < imageOneData.keyPointDescriptors.rows; i++) {
        double distance = matchedFeatures[i].distance;
        if (distance < minDistance) {
            minDistance = distance;
        }

        if (distance > maxDistance) {
            maxDistance = distance;
        }

        if (matchedFeatures[i].distance <= std::max(2 * minDistance, 0.02)) {
            goodMatches.push_back(matchedFeatures[i]);
        }
    }

    std::cout << "Maximum distance in matched features: " << maxDistance << std::endl;
    std::cout << "Minimum distance in matched features: " << minDistance << std::endl;

    cv::Mat imageOfMatches;
    cv::drawMatches(imageOneData.imageData, imageOneData.keyPoints,
                    imageTwoData.imageData, imageTwoData.keyPoints,
                    goodMatches, imageOfMatches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Image with matches", imageOfMatches);

    std::cout << "Good matches found:" << std::endl;
    for (auto feature : goodMatches) {
        std::cout << "KeyPoint 1 " << feature.queryIdx << ", KeyPoint 2 " << feature.trainIdx << std::endl;
    }

    cv::waitKey(0);
}

void ChangeDetector::FlannMatcher::match(cv::Mat &pDescriptorOne, cv::Mat &pDescriptorTwo) {
    cv::FlannBasedMatcher flannMatcher;

    flannMatcher.match(pDescriptorOne, pDescriptorTwo, matchedFeatures);
}

