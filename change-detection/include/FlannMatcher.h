//
// Created by Harpreet Singh on 10/16/17.
//

#ifndef CHANGE_DETECTION_FLANNMATCHER_H
#define CHANGE_DETECTION_FLANNMATCHER_H

#include <iostream>
#include <math.h>
#include <functional>
#include <utility>
#include <opencv2/xfeatures2d.hpp>
#include "FeatureMatch.h"

namespace ChangeDetector {
    class FlannMatcher : public FeatureMatch {
    public:
        FlannMatcher() {}

        ~FlannMatcher() {}

        std::vector<cv::DMatch> getMatchedFeatures(std::function<bool(cv::DMatch &, std::pair<double, double>)> &pCriteria);

        std::pair<double, double> getMinAndMaxFeatureMatchDistances();

    protected:
        std::vector<cv::KeyPoint> featureDetection(cv::Mat &pImage);

        cv::Mat featureDescription(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures);

        void preProcess();

        void match(cv::Mat &pDescriptorOne, cv::Mat &pDescriptorTwo);

        void postProcess();

    private:
        std::vector<cv::DMatch> matchedFeatures;
        double minDistanceOfMatches;
        double maxDistanceOfMatches;
    };
}


#endif //CHANGE_DETECTION_FLANNMATCHER_H
