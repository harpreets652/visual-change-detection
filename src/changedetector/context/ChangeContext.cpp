//
// Created by Harpreet Singh on 10/28/17.
//

#include "ChangeContext.h"

ChangeDetector::ChangeContext::ChangeContext(DataVector pQueryDataPaths, DataVector pReferenceDataPaths) {
    if (pQueryDataPaths.empty() || pReferenceDataPaths.empty()) {
        throw std::invalid_argument("Query and reference data must be specified");
    }

    std::pair<std::string, std::string> &primaryImageData = pQueryDataPaths.front();
    queryData.primaryImage = readImageData(primaryImageData.first, primaryImageData.second);

    for (int i = 1; i < pQueryDataPaths.size(); i++) {
        std::pair<std::string, std::string> &dataPaths = pQueryDataPaths[i];
        ImageData imageData = readImageData(dataPaths.first, dataPaths.second);
        queryData.supplementalImages.push_back(imageData);
    }

    primaryImageData = pReferenceDataPaths.front();
    referenceData.primaryImage = readImageData(primaryImageData.first, primaryImageData.second);

    for (int i = 1; i < pReferenceDataPaths.size(); i++) {
        std::pair<std::string, std::string> &dataPaths = pReferenceDataPaths[i];
        ImageData imageData = readImageData(dataPaths.first, dataPaths.second);
        referenceData.supplementalImages.push_back(imageData);
    }
}

ChangeDetector::ImageData
ChangeDetector::ChangeContext::readImageData(std::string &pImagePath, std::string &pRobotPose) {
    //read image
    cv::Mat image = cv::imread(pImagePath);
    if (image.empty()) {
        throw std::runtime_error("Unable to read image " + pImagePath);
    }

    //read pose
    Pose imagePose = ChangeDetector::Pose();
    std::ifstream poseFile(pRobotPose);
    if (!poseFile.good()) {
        throw std::runtime_error("Unable to read pose data " + pRobotPose);
    }

    std::string dummyLine, poseData, dataPoint;
    std::getline(poseFile, dummyLine);
    std::getline(poseFile, poseData);

    std::stringstream poseDataStream(poseData);

    //ignore message seq and timestamp at the beginning
    std::getline(poseDataStream, dataPoint, ',');
    std::getline(poseDataStream, dataPoint, ',');

    //position x, y, and z
    for (int i = 0; i < 3; i++) {
        std::getline(poseDataStream, dataPoint, ',');
        imagePose.position[i] = std::stod(dataPoint);
    }

    //orientation x, y, z, and w
    for (int i = 0; i < 4; i++) {
        std::getline(poseDataStream, dataPoint, ',');
        imagePose.orientation[i] = std::stod(dataPoint);
    }

    ImageData imageData;
    imageData.image = image;
    imageData.robotPose = imagePose;

    return imageData;
}

ChangeDetector::InputData &ChangeDetector::ChangeContext::getQueryImageData() {
    return queryData;
}

ChangeDetector::InputData &ChangeDetector::ChangeContext::getReferenceImageData() {
    return referenceData;
}