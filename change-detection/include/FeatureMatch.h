//
// Created by Harpreet Singh on 10/16/17.
//

#ifndef CHANGE_DETECTION_FEATUREMATCH_H
#define CHANGE_DETECTION_FEATUREMATCH_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Detector.h"
#include "Descriptor.h"

namespace ChangeDetector {
    class FeatureMatch {
    public:
        FeatureMatch();

        virtual ~FeatureMatch() {}

        FeatureMatch *withDetector(Detector *pFeatureDetector);

        FeatureMatch *withDescriptor(Descriptor *pFeatureDescriptor);

        void execute(std::string pImage1, std::string pImage2);

    protected:
        //************override these methods*************************
        virtual void preProcess() {};

        virtual void match(cv::Mat &pDescriptorOne, cv::Mat &pDescriptorTwo) {};

        virtual void postProcess() {};
        //************override these methods*************************

        struct ImageDataContainer {
            std::string imageName;
            cv::Mat imageData;
            std::vector<cv::KeyPoint> keyPoints;
            cv::Mat keyPointDescriptors;
        };

        Detector *featureDetector;
        Descriptor *featureDescriptor;

        ImageDataContainer imageOneData;
        ImageDataContainer imageTwoData;
    };
}


#endif //CHANGE_DETECTION_FEATUREMATCH_H
