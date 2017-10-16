//
// Created by Harpreet Singh on 10/16/17.
//

#ifndef CHANGE_DETECTION_FLANNMATCHER_H
#define CHANGE_DETECTION_FLANNMATCHER_H

#include <iostream>
#include <math.h>
#include <opencv2/xfeatures2d.hpp>
#include "FeatureMatch.h"

namespace ChangeDetector {
    class FlannMatcher : public FeatureMatch {
    public:
        FlannMatcher() {}

        ~FlannMatcher() {}

    protected:
        void preProcess();

        void match(cv::Mat &pDescriptorOne, cv::Mat &pDescriptorTwo);

        void postProcess();

    private:
        std::vector<cv::DMatch> matchedFeatures;
    };
}


#endif //CHANGE_DETECTION_FLANNMATCHER_H
