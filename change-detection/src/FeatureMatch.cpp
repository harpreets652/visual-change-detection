//
// Created by Harpreet Singh on 10/16/17.
//

#include "FeatureMatch.h"

ChangeDetector::FeatureMatch::FeatureMatch() {
    featureDetector = nullptr;
    featureDescriptor = nullptr;
}

ChangeDetector::FeatureMatch *ChangeDetector::FeatureMatch::withDetector(Detector *pFeatureDetector) {
    featureDetector = pFeatureDetector;
    return this;
}

ChangeDetector::FeatureMatch *ChangeDetector::FeatureMatch::withDescriptor(Descriptor *pFeatureDescriptor) {
    featureDescriptor = pFeatureDescriptor;
    return this;
}


void ChangeDetector::FeatureMatch::execute(std::string pImage1, std::string pImage2) {
    imageOneData.imageName = std::move(pImage1);
    imageTwoData.imageName = std::move(pImage2);

    imageOneData.imageData = cv::imread(imageOneData.imageName, cv::IMREAD_GRAYSCALE);
    if (imageOneData.imageData.empty()) {
        std::cerr << "Unable to open input image " << imageOneData.imageName << std::endl;
        return;
    }

    imageTwoData.imageData = cv::imread(imageTwoData.imageName, cv::IMREAD_GRAYSCALE);
    if (imageTwoData.imageData.empty()) {
        std::cerr << "Unable to open input image " << imageTwoData.imageName << std::endl;
        return;
    }

    std::cout << "Running feature detector" << std::endl;
    if (featureDetector == nullptr || featureDescriptor == nullptr) {
        std::cerr << "Feature detector and descriptor must be specified" << std::endl;
        return;
    }

    imageOneData.keyPoints = featureDetector->getFeatures(imageOneData.imageData);
    imageTwoData.keyPoints = featureDetector->getFeatures(imageTwoData.imageData);

    std::cout << "Running feature descriptor" << std::endl;
    imageOneData.keyPointDescriptors = featureDescriptor->getDescriptors(imageOneData.imageData,
                                                                         imageOneData.keyPoints);
    imageTwoData.keyPointDescriptors = featureDescriptor->getDescriptors(imageTwoData.imageData,
                                                                         imageTwoData.keyPoints);

    std::cout << "pre-process step" << std::endl;
    preProcess();

    std::cout << "Match features" << std::endl;
    match(imageOneData.keyPointDescriptors, imageTwoData.keyPointDescriptors);

    std::cout << "post-process step" << std::endl;
    postProcess();
}