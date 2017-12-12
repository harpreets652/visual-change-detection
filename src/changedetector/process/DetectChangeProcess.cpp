//
// Created by Harpreet Singh on 11/19/17.
//

#include "DetectChangeProcess.h"

static const bool DEBUGGING = true;

ChangeDetector::DetectChangeProcess::DetectChangeProcess(ChangeDetector::ChangeContext *pChangeContext) {
    if (pChangeContext == nullptr) {
        throw std::invalid_argument("Change Context must be specified");
    }

    changeContextPtr = pChangeContext;
}


cv::Mat ChangeDetector::DetectChangeProcess::execute() {
    //find bounding boxes, generate masks for each image
    auto imageData = changeContextPtr->getQueryImageData();

//    filtering(imageData.primaryImage.image);

    auto queryImageSaliency = getSaliency(imageData.primaryImage.image);
    convertToOneChannelEightBit(queryImageSaliency);

    averageAcrossSupplementaryImages(queryImageSaliency, imageData);

    auto queryImageThreshold = thresholdImage(queryImageSaliency);

    if (DEBUGGING) {
        //Note~ debugging code
        showImage(changeContextPtr->getQueryImageData().primaryImage.image, "query image");
        showImage(queryImageSaliency, "query image saliency");
        showImage(queryImageThreshold, "query image threshold");
    }

    std::vector<cv::Rect> queryImageRegions = findBoundingBoxes(queryImageThreshold);

    cv::Mat queryImageMask = constructMask(queryImageRegions,
                                           changeContextPtr->getQueryImageData().primaryImage.image.size());

    //generate features and descriptors
    std::vector<cv::KeyPoint> queryKeyPoints = changeContextPtr->getKeyPoints(
            changeContextPtr->getQueryImageData().primaryImage.image,
            queryImageMask);

    cv::Mat referenceImageMask;
    std::vector<cv::KeyPoint> referenceKeyPoints = changeContextPtr->getKeyPoints(
            changeContextPtr->getReferenceImageData().primaryImage.image,
            referenceImageMask);

    if (DEBUGGING) {
        //Note~ debugging code
        cv::Mat queryTempImage;
        cv::drawKeypoints(changeContextPtr->getQueryImageData().primaryImage.image,
                          queryKeyPoints,
                          queryTempImage);

        for (const auto &rect : queryImageRegions) {
            cv::rectangle(queryTempImage, rect, cv::Scalar(0, 0, 255), 5);
        }
        showImage(queryTempImage, "query image, features, salient regions");
        queryTempImage.release();

        cv::Mat referenceTempImage;
        cv::drawKeypoints(changeContextPtr->getReferenceImageData().primaryImage.image,
                          referenceKeyPoints,
                          referenceTempImage);
        showImage(referenceTempImage, "reference image, features, salient regions");
        referenceTempImage.release();
    }

    cv::Mat queryDescriptors = changeContextPtr->getKeyPointDescriptors(
            changeContextPtr->getQueryImageData().primaryImage.image,
            queryKeyPoints);
    cv::Mat referenceDescriptors = changeContextPtr->getKeyPointDescriptors(
            changeContextPtr->getReferenceImageData().primaryImage.image,
            referenceKeyPoints);
    //identify regions of change Note~ should call this with query -> ref and vice versa
    std::vector<cv::Rect> regionsOfDifference = findRegionsOfDifference(queryImageRegions,
                                                                        queryKeyPoints,
                                                                        queryDescriptors,
                                                                        referenceDescriptors,
                                                                        referenceKeyPoints);
    if (!regionsOfDifference.empty()) {
        cv::Mat changeMask = constructMask(regionsOfDifference,
                                           changeContextPtr->getQueryImageData().primaryImage.image.size());
        cv::Mat changesImage;
        changeContextPtr->getQueryImageData().primaryImage.image.copyTo(changesImage, changeMask);

//        extractObject(regionsOfDifference, changeContextPtr->getQueryImage());

        return changesImage;
    } else {
        std::cout << "No differences found" << std::endl;
        return cv::Mat();
    }
}

void ChangeDetector::DetectChangeProcess::filtering(cv::Mat &pInputImage) {
    float boundaryFilter[3][3] = {{2, 2,   2},
                                  {2, -15, 2},
                                  {2, 2,   2}};

    cv::Mat filter(3, 3, CV_32FC1, &boundaryFilter);
    filter2D(pInputImage, pInputImage, -1, filter);

    if (DEBUGGING) {
        showImage(pInputImage, "filtered image");
    }
}

void ChangeDetector::DetectChangeProcess::extractObject(std::vector<cv::Rect> &pRegions, cv::Mat &pImage) {
    cv::Mat imageClone;
    cv::cvtColor(pImage, imageClone, CV_GRAY2BGR);

    for (auto &rect : pRegions) {
        cv::Rect area = rect;
        area.height *= 2;
        area.width *= 2;
        int newCenterX = (area.x + area.width) / 2;
        int newCenterY = (area.y + area.height) / 2;
        int oldCenterX = (rect.x + rect.width) / 2;
        int oldCenterY = (rect.y + rect.height) / 2;

        area.x -= (newCenterX - oldCenterX) / 2;
        area.y -= (newCenterY - oldCenterY) / 2;

        if (area.x + area.width > pImage.size().width) {
            area.width = pImage.size().width - area.x;
        }

        if (area.y + area.height > pImage.size().height) {
            area.height = pImage.size().height - area.y;
        }

        if (true) {
            cv::Mat queryTempImage = pImage.clone();
            cv::rectangle(queryTempImage, area, cv::Scalar(0, 0, 255), 5);
            showImage(queryTempImage, "rect region now");
        }

        cv::Mat bgModel;
        cv::Mat fgModel;
        cv::Mat result;

        cv::grabCut(imageClone, result, area, bgModel, fgModel, 1, cv::GC_INIT_WITH_RECT);
        cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);

        showImage(result, "bla bla");

        cv::Mat foreground(imageClone.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        imageClone.copyTo(foreground, result);
        showImage(foreground, "testing");
    }
}

std::vector<cv::Rect>
ChangeDetector::DetectChangeProcess::findRegionsOfDifference(std::vector<cv::Rect> &pQueryRegions,
                                                             std::vector<cv::KeyPoint> &pQueryKeyPoints,
                                                             cv::Mat &pQueryDescriptors,
                                                             cv::Mat &pReferenceDescriptors,
                                                             std::vector<cv::KeyPoint> &pReferenceKeyPoints) {
    std::vector<int> potentialRegionOfChange;
    for (int i = 0; i < pQueryRegions.size(); i++) {
        cv::Rect region = pQueryRegions[i];

        //find features within bounds
        std::vector<int> keyPointsWithinBounds = getKeyPointIndicesInRegion(pQueryKeyPoints, region);
        if (keyPointsWithinBounds.empty() || keyPointsWithinBounds.size() < 5) {
            std::cout << "No key points within bounds, skipping region" << std::endl;
            continue;
        }

        if (DEBUGGING) {
            cv::Mat boundsKeyPointImage;
            cv::drawKeypoints(changeContextPtr->getQueryImageData().primaryImage.image,
                              getKeyPointsInRegion(pQueryKeyPoints, region),
                              boundsKeyPointImage);
            cv::rectangle(boundsKeyPointImage, region, cv::Scalar(0, 0, 255), 5);

            showImage(boundsKeyPointImage, "Key points in query image");
        }

        std::vector<cv::DMatch> matches = getGoodMatches(keyPointsWithinBounds,
                                                         pQueryDescriptors,
                                                         pReferenceDescriptors);
        if (matches.empty()) {
            std::cout << "No good matches, continuing" << std::endl;
            continue;
        }

        auto percentMatches = ((float) matches.size() / (float) keyPointsWithinBounds.size()) * 100.0;
        if (percentMatches <= 80) {
            potentialRegionOfChange.push_back(i);
        }

        if (DEBUGGING) {
            std::cout << "Percent match: " << percentMatches << std::endl;
            cv::Mat debugImage;
            cv::drawMatches(changeContextPtr->getQueryImageData().primaryImage.image,
                            getKeyPointsInRegion(pQueryKeyPoints, region),
                            changeContextPtr->getReferenceImageData().primaryImage.image,
                            pReferenceKeyPoints,
                            matches,
                            debugImage,
                            cv::Scalar::all(-1),
                            cv::Scalar::all(-1), std::vector<char>(),
                            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            showImage(debugImage, "query to reference matches");
        }
    }

    std::vector<cv::Rect> pRegions;
    for (auto i: potentialRegionOfChange) {
        pRegions.push_back(pQueryRegions[i]);
    }

    return pRegions;
}

std::vector<cv::DMatch> ChangeDetector::DetectChangeProcess::getGoodMatches(std::vector<int> &pKeyPointIndices,
                                                                            cv::Mat &pQueryDescriptors,
                                                                            cv::Mat &pReferenceDescriptors) {
    cv::Mat subsetDescriptors = filterDescriptorsByKeyPointIndices(pKeyPointIndices, pQueryDescriptors);

    std::vector<cv::DMatch> goodMatches;
    if (subsetDescriptors.empty()) {
        return goodMatches;
    }

    std::vector<std::vector<cv::DMatch>> kNearestMatches = changeContextPtr->matchFeaturesKNN(subsetDescriptors,
                                                                                              pReferenceDescriptors,
                                                                                              2);

    const float thresholdRatio = 0.95;
    for (const auto &kMatches : kNearestMatches) {
        if (kMatches[0].distance < thresholdRatio * kMatches[1].distance) {
            goodMatches.push_back(kMatches[0]);
        }
    }

    return goodMatches;
}

cv::Mat ChangeDetector::DetectChangeProcess::filterDescriptorsByKeyPointIndices(std::vector<int> &pKeyPointIndices,
                                                                                cv::Mat pDescriptors) {
    cv::Mat subsetDescriptors(static_cast<int>(pKeyPointIndices.size()),
                              pDescriptors.cols,
                              pDescriptors.type());

    for (int i = 0; i < pKeyPointIndices.size(); i++) {
        if (pKeyPointIndices[i] < pDescriptors.rows) {
            pDescriptors.row(pKeyPointIndices[i]).copyTo(subsetDescriptors.row(i));
        }
    }

    return subsetDescriptors;
}

std::vector<int> ChangeDetector::DetectChangeProcess::getKeyPointIndicesInRegion(std::vector<cv::KeyPoint> &pKeyPoints,
                                                                                 cv::Rect &pRegion) {
    std::vector<int> indices;
    for (int i = 0; i < pKeyPoints.size(); i++) {
        if (pRegion.contains(pKeyPoints[i].pt)) {
            indices.push_back(i);
        }
    }

    return indices;
}

std::vector<cv::KeyPoint>
ChangeDetector::DetectChangeProcess::getKeyPointsInRegion(std::vector<cv::KeyPoint> &pKeyPoints,
                                                          cv::Rect &pRegion) {
    std::vector<cv::KeyPoint> points;

    for (auto keyPoint : pKeyPoints) {
        if (pRegion.contains(keyPoint.pt)) {
            points.push_back(keyPoint);
        }
    }

    return points;
}

cv::Mat ChangeDetector::DetectChangeProcess::constructMask(std::vector<cv::Rect> pRegions, const cv::Size &pSize) {
    cv::Mat mask = cv::Mat::zeros(pSize, CV_8UC1);
    for (const auto &box : pRegions) {
        mask(box) = 1;
    }

    return mask;
}

std::vector<cv::Rect> ChangeDetector::DetectChangeProcess::findBoundingBoxes(cv::Mat &pBinaryImage) {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(pBinaryImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    std::vector<cv::Rect> boundingRectangles;
    for (const auto &contourList : contours) {
        boundingRectangles.push_back(cv::boundingRect(contourList));
    }

    return boundingRectangles;
}

cv::Mat ChangeDetector::DetectChangeProcess::thresholdImage(cv::Mat &pInputImage) {
    cv::Mat binaryImage;
    cv::threshold(pInputImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    return binaryImage;
}

void ChangeDetector::DetectChangeProcess::averageAcrossSupplementaryImages(cv::Mat &queryImageSaliency,
                                                                           InputData imageData) {
    const int numberOfImages = imageData.supplementalImages.size() + 1;
    for (auto imageInfo : imageData.supplementalImages) {

//        filtering(imageInfo.image);

        auto supplementaryImageSaliency = getSaliency(imageInfo.image);
        convertToOneChannelEightBit(supplementaryImageSaliency);
        add(queryImageSaliency, supplementaryImageSaliency, queryImageSaliency);
    }
    queryImageSaliency = queryImageSaliency / numberOfImages;
}

void ChangeDetector::DetectChangeProcess::convertToOneChannelEightBit(cv::Mat &pImage) {
    double min, max;
    cv::minMaxLoc(pImage, &min, &max);
    pImage.convertTo(pImage, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));
}
