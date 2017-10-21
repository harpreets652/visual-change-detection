//
// Created by Harpreet Singh on 10/16/17.
//

#include "FeatureMatch.h"

ChangeDetector::FeatureMatch::FeatureMatch() {
}

void ChangeDetector::FeatureMatch::execute(std::string pImage1, std::string pImage2) {
    queryImage.imageName = std::move(pImage1);
    trainingImage.imageName = std::move(pImage2);

    queryImage.imageData = cv::imread(queryImage.imageName, cv::IMREAD_GRAYSCALE);
    if (queryImage.imageData.empty()) {
        std::cerr << "Unable to open input image " << queryImage.imageName << std::endl;
        return;
    }

    trainingImage.imageData = cv::imread(trainingImage.imageName, cv::IMREAD_GRAYSCALE);
    if (trainingImage.imageData.empty()) {
        std::cerr << "Unable to open input image " << trainingImage.imageName << std::endl;
        return;
    }

    std::cout << "Running feature detector" << std::endl;
    queryImage.keyPoints = featureDetection(queryImage.imageData);
    trainingImage.keyPoints = featureDetection(trainingImage.imageData);

    if (queryImage.keyPoints.empty() || trainingImage.keyPoints.empty()) {
        std::cerr << "Feature detector did not find features." << std::endl;
        return;
    }

    std::cout << "Running feature descriptor" << std::endl;
    queryImage.keyPointDescriptors = featureDescription(queryImage.imageData,
                                                          queryImage.keyPoints);

    trainingImage.keyPointDescriptors = featureDescription(trainingImage.imageData,
                                                          trainingImage.keyPoints);

    if (queryImage.keyPointDescriptors.empty() || trainingImage.keyPointDescriptors.empty()) {
        std::cerr << "Feature description returned empty results." << std::endl;
        return;
    }

    std::cout << "pre-process step" << std::endl;
    preProcess();

    std::cout << "Match features" << std::endl;
    match(queryImage.keyPointDescriptors, trainingImage.keyPointDescriptors);

    std::cout << "post-process step" << std::endl;
    postProcess();
}

ChangeDetector::ImageDataContainer *ChangeDetector::FeatureMatch::getQueryImageData() {
    return &queryImage;
}

ChangeDetector::ImageDataContainer *ChangeDetector::FeatureMatch::getTrainingImageData() {
    return &trainingImage;
}