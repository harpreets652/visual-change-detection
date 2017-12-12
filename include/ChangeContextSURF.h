//
// Created by Harpreet Singh on 10/28/17.
//

#ifndef CHANGE_DETECTION_CHANGECONTEXTSURF_H
#define CHANGE_DETECTION_CHANGECONTEXTSURF_H

#include "ChangeContext.h"
#include <opencv2/xfeatures2d.hpp>

namespace ChangeDetector {
    class ChangeContextSURF : public ChangeContext {
    public:
        ChangeContextSURF(DataVector pQueryDataPaths, DataVector pReferenceDataPaths) :
                ChangeContext(pQueryDataPaths, pReferenceDataPaths) {}

        ~ChangeContextSURF() {}

        std::vector<cv::KeyPoint> getKeyPoints(cv::Mat &pImage, cv::Mat &pMask);

        cv::Mat getKeyPointDescriptors(cv::Mat &pImage, std::vector<cv::KeyPoint> pKeyPoints);

        std::vector<std::vector<cv::DMatch>>
        matchFeaturesKNN(cv::Mat &pQueryDescriptors, cv::Mat &pReferenceDescriptors, int pK);
    };
}

#endif //CHANGE_DETECTION_CHANGECONTEXTSURF_H
