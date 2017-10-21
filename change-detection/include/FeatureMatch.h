//
// Created by Harpreet Singh on 10/16/17.
//

#ifndef CHANGE_DETECTION_FEATUREMATCH_H
#define CHANGE_DETECTION_FEATUREMATCH_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace ChangeDetector {
    struct ImageDataContainer {
        std::string imageName;
        cv::Mat imageData;
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat keyPointDescriptors;
    };

    class FeatureMatch {
    public:
        FeatureMatch();

        virtual ~FeatureMatch() {}

        void execute(std::string pImage1, std::string pImage2);

        ImageDataContainer *getQueryImageData();

        ImageDataContainer *getTrainingImageData();

    protected:
        //************override these methods*************************
        virtual std::vector<cv::KeyPoint> featureDetection(cv::Mat &pImage)=0;

        virtual cv::Mat featureDescription(cv::Mat &pImage, std::vector<cv::KeyPoint> pFeatures)=0;

        virtual void preProcess()=0;

        virtual void match(cv::Mat &pDescriptorOne, cv::Mat &pDescriptorTwo)=0;

        virtual void postProcess()=0;
        //************override these methods*************************

        ImageDataContainer queryImage;
        ImageDataContainer trainingImage;
    };
}


#endif //CHANGE_DETECTION_FEATUREMATCH_H
