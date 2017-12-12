//
// Created by Harpreet Singh on 11/19/17.
//

#ifndef CHANGE_DETECTION_DETECTCHANGEPROCESS_H
#define CHANGE_DETECTION_DETECTCHANGEPROCESS_H

#include <iostream>
#include <memory>
#include "ChangeContext.h"
#include "Utils.h"

namespace ChangeDetector {
    class DetectChangeProcess {
    public:
        DetectChangeProcess(ChangeDetector::ChangeContext *pChangeContext);

        ~DetectChangeProcess() {}

        cv::Mat execute();

    private:
        void extractObject(std::vector<cv::Rect> &pRegions, cv::Mat &pImage);

        void convertToOneChannelEightBit(cv::Mat &pImage);

        cv::Mat thresholdImage(cv::Mat &pInputImage);

		void averageAcrossSupplementaryImages(cv::Mat &queryImageSaliency, InputData imageData);

        std::vector<cv::Rect> findBoundingBoxes(cv::Mat &pBinaryImage);

        cv::Mat constructMask(std::vector<cv::Rect> pRegions, const cv::Size &pSize);

        std::vector<int> getKeyPointIndicesInRegion(std::vector<cv::KeyPoint> &pKeyPoints, cv::Rect &pRegion);

        std::vector<cv::KeyPoint> getKeyPointsInRegion(std::vector<cv::KeyPoint> &pKeyPoints, cv::Rect &pRegion);

        cv::Mat filterDescriptorsByKeyPointIndices(std::vector<int> &pKeyPointIndices, cv::Mat pDescriptors);

        void filtering(cv::Mat &pInputImage);


        std::vector<cv::DMatch> getGoodMatches(std::vector<int> &pKeyPointIndices,
                                               cv::Mat &pQueryDescriptors,
                                               cv::Mat &pReferenceDescriptors);

        std::vector<cv::Rect> findRegionsOfDifference(std::vector<cv::Rect> &pQueryRegions,
                                                      std::vector<cv::KeyPoint> &pQueryKeyPoints,
                                                      cv::Mat &pQueryDescriptors,
                                                      cv::Mat &pReferenceDescriptors,
                                                      std::vector<cv::KeyPoint> &pReferenceKeyPoints);

        ChangeDetector::ChangeContext *changeContextPtr;
    };
}


#endif //CHANGE_DETECTION_DETECTCHANGEPROCESS_H
