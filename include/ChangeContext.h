//
// Created by Harpreet Singh on 10/28/17.
//

#ifndef CHANGE_DETECTION_CHANGECONTEXT_H
#define CHANGE_DETECTION_CHANGECONTEXT_H

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

namespace ChangeDetector {
    typedef std::vector<std::pair<std::string, std::string>> DataVector;

    struct Pose {
        //elements 0, 1, 2 is x, y, and z respectively
        cv::Vec3d position;

        //elements 0, 1, 2, 3 is x, y, z, and w respectively
        cv::Vec4d orientation;
    };

    struct ImageData {
        cv::Mat image;
        Pose robotPose;
    };

    struct InputData {
        ImageData primaryImage;
        std::vector<ImageData> supplementalImages;
    };

    class ChangeContext {
    public:
        ChangeContext() {};

        ChangeContext(DataVector pQueryDataPaths, DataVector pReferenceDataPaths);

        virtual ~ChangeContext() {}

        InputData &getQueryImageData();

        InputData &getReferenceImageData();

        //************implement this method*************************
        virtual std::vector<cv::KeyPoint> getKeyPoints(cv::Mat &pImage, cv::Mat &pMask)=0;

        virtual cv::Mat getKeyPointDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pKeyPoints)=0;

        virtual std::vector<std::vector<cv::DMatch>>
        matchFeaturesKNN(cv::Mat &pQueryDescriptors, cv::Mat &pReferenceDescriptors, int pK) = 0;
        //************implement this method*************************

    protected:
        ImageData readImageData(std::string &pImagePath, std::string &pRobotPose);

        InputData queryData;
        InputData referenceData;
    };
}


#endif //CHANGE_DETECTION_CHANGECONTEXT_H
