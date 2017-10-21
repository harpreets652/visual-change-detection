#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "FeatureMatch.h"
#include "FlannMatcher.h"

void showImage(std::string &imageName);

void showImage(cv::Mat &image);

void featureDetectorSURF(std::string &imageName);

void featureMatchUsingFlann(std::string &queryImage, std::string &trainImage);

int main() {
    std::string changedImage = "../resources/Mission2.JPG";
    std::string originalImage = "../resources/Mission1.JPG";

    //feature matching between two images
    featureMatchUsingFlann(changedImage, originalImage);

    //todo: use features that are matching to determine if the images are the same

    //detect groups of features not matching
    //Note~ k means clustering on features that are not good matches?
    //Note~ will also need to detect things that have been removed...change api of matcher to be matchAtoB, matchBtoA

    return 0;
}

void featureMatchUsingFlann(std::string &queryImage, std::string &trainImage) {
    //feature matching on two images
    auto *featureMatcher = new ChangeDetector::FlannMatcher();
    featureMatcher->execute(queryImage, trainImage);

    //select features that did not match and show them on the query image
    std::function<bool(cv::DMatch &, std::pair<double, double>)> selectCriteria = [](cv::DMatch &pMatch,
                                                                                     std::pair<double, double> minMax) {
        return pMatch.distance >= std::max(2 * minMax.first, 0.1);
    };
    std::vector<cv::DMatch> matchingDescriptors = featureMatcher->getMatchedFeatures(selectCriteria);

    std::vector<cv::KeyPoint> correspondingFeatures;
    for (auto match : matchingDescriptors) {
        correspondingFeatures.push_back(featureMatcher->getQueryImageData()->keyPoints[match.queryIdx]);
    }
    cv::Mat matchingFeaturesImage;
    cv::drawKeypoints(featureMatcher->getQueryImageData()->imageData, correspondingFeatures, matchingFeaturesImage);

    showImage(matchingFeaturesImage);
}

void featureDetectorSURF(std::string &imageName) {
    cv::Mat inputImage = cv::imread(imageName, cv::IMREAD_GRAYSCALE);

    if (!inputImage.data) {
        std::cerr << "Unable to open input image " << imageName << std::endl;
        return;
    }

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

    std::vector<cv::KeyPoint> keyPoints;
    detector->detect(inputImage, keyPoints);

    cv::Mat keyPointsImage;
    cv::drawKeypoints(inputImage, keyPoints, keyPointsImage);

    showImage(keyPointsImage);
}

void showImage(std::string &imageName) {
    cv::Mat testing = cv::imread(imageName);

    if (!testing.data) {
        std::cerr << "unable to read image" << std::endl;
    }

    showImage(testing);
}

void showImage(cv::Mat &image) {
    cv::namedWindow("image", cv::WINDOW_KEEPRATIO);
    cv::imshow("image", image);

    cv::waitKey(0);
}