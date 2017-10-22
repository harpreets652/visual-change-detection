#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "FeatureMatch.h"
#include "FlannMatcher.h"

void showImage(std::string &imageName);

void showImage(cv::Mat &image);

void featureDetectorSURF(std::string &imageName);

void featureMatchUsingFlann(std::string &queryImage, std::string &trainImage);

void findClusters(const std::vector<cv::KeyPoint> &pFeatures, ChangeDetector::FlannMatcher *matcher);

int main() {
    std::string changedImage = "../resources/Mission2.JPG";
    std::string originalImage = "../resources/Mission1.JPG";

    //feature matching between two images
    featureMatchUsingFlann(changedImage, originalImage);

    //todo: use features that are matching to determine if the images are the same

    //detect groups of features not matching
    //Note~ will also need to detect things that have been removed...change api of matcher to be matchAtoB, matchBtoA

    return 0;
}

void featureMatchUsingFlann(std::string &queryImage, std::string &trainImage) {
    //feature matching on two images
    auto *featureMatcher = new ChangeDetector::FlannMatcher();
    featureMatcher->execute(queryImage, trainImage);

    //select descriptors that did not match
    std::function<bool(cv::DMatch &, std::pair<double, double>)> selectCriteria = [](cv::DMatch &pMatch,
                                                                                     std::pair<double, double> minMax) {
        return pMatch.distance >= std::max(2 * minMax.first, 0.1);
    };
    std::vector<cv::DMatch> matchingDescriptors = featureMatcher->getMatchedFeatures(selectCriteria);

    //find corresponding features of the non-matching descriptors
    std::vector<cv::KeyPoint> correspondingFeatures;
    for (auto match : matchingDescriptors) {
        correspondingFeatures.push_back(featureMatcher->getQueryImageData()->keyPoints[match.queryIdx]);
    }

    //look for clusters of unmatched features
    findClusters(correspondingFeatures, featureMatcher);

    //the above features
//    cv::Mat matchingFeaturesImage;
//    cv::drawKeypoints(featureMatcher->getQueryImageData()->imageData, correspondingFeatures, matchingFeaturesImage);

//    showImage(matchingFeaturesImage);
}

void findClusters(const std::vector<cv::KeyPoint> &pFeatures, ChangeDetector::FlannMatcher *matcher) {
    cv::Mat inputData((int) pFeatures.size(), 2, CV_32F, cv::Scalar::all(0));

    for (int i = 0; i < inputData.rows; i++) {
        inputData.at<float>(i, 0) = pFeatures[i].pt.x;
        inputData.at<float>(i, 1) = pFeatures[i].pt.y;
    }

    cv::Scalar colorTab[] = {
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 100, 100),
            cv::Scalar(255, 0, 255),
            cv::Scalar(0, 255, 255)
    };

    cv::Mat labels, centers;
    double compactness = cv::kmeans(inputData,
                                    3,
                                    labels,
                                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 1.0),
                                    1,
                                    cv::KMEANS_PP_CENTERS,
                                    centers);

    //draw the clusters
    cv::Mat clustersImage;
    cv::cvtColor(matcher->getQueryImageData()->imageData, clustersImage, CV_GRAY2BGR);

    for (int k = 0; k < inputData.rows; k++) {
        int clusterIdentifier = labels.at<int>(k);
        cv::Point2f featurePt = inputData.at<cv::Point2f>(k);
        cv::circle(clustersImage, featurePt, 2, colorTab[clusterIdentifier], cv::FILLED, cv::LINE_AA);
    }

    for (int j = 0; j < centers.rows; j++) {
        cv::Point2f centerPoint = centers.at<cv::Point2f>(j);
        cv::circle(clustersImage, centerPoint, 5, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }

    showImage(clustersImage);
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