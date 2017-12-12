//
// Created by Harpreet Singh on 11/1/17.
//

#ifndef CHANGE_DETECTION_UTILS_H
#define CHANGE_DETECTION_UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>

void showImage(cv::Mat &image, std::string pTitle="DefaultTitle");

void showImage(std::string &imageName, std::string &pTitle);

cv::Mat getSaliency(cv::Mat &pImage);

#endif //CHANGE_DETECTION_UTILS_H
