#include <iostream>
#include <opencv2/opencv.hpp>


void showImage(std::string &imageName);

int main() {

    std::string imageName = "../resources/chessboard-texture.png";
    showImage(imageName);

    return 0;
}

void showImage(std::string &imageName){
    cv::Mat testing = cv::imread(imageName);

    if (!testing.data) {
        std::cerr << "unable to read image" << std::endl;
    }

    cv::namedWindow("testing open cv", cv::WINDOW_AUTOSIZE);
    cv::imshow("image name", testing);

    cv::waitKey(0);
}