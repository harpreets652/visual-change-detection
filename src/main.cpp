#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "ChangeContextSURF.h"
#include "ChangeContextBinary.h"
#include "DetectChangeProcess.h"


void changeDetection(ChangeDetector::DataVector &queryImage, ChangeDetector::DataVector &trainImage);

int main() {
/*
    ChangeDetector::DataVector queryData = {
            std::make_pair("../resources/mission2/color/set1/651_1508459484.901719006.jpg",
                           "../resources/mission2/color/set1/651_1508459484.901719006.txt"),
            std::make_pair("../resources/mission2/color/set1/652_1508459484.944026033.jpg",
                           "../resources/mission2/color/set1/652_1508459484.944026033.txt"),
            std::make_pair("../resources/mission2/color/set1/648_1508459484.678660354.jpg",
                           "../resources/mission2/color/set1/648_1508459484.678660354.txt"),
            std::make_pair("../resources/mission2/color/set1/649_1508459484.804956132.jpg",
                           "../resources/mission2/color/set1/649_1508459484.804956132.txt"),
            std::make_pair("../resources/mission2/color/set1/653_1508459485.47180631.jpg",
                           "../resources/mission2/color/set1/653_1508459485.47180631.txt"),
            std::make_pair("../resources/mission2/color/set1/654_1508459485.101761933.jpg",
                           "../resources/mission2/color/set1/654_1508459485.101761933.txt"),
            std::make_pair("../resources/mission2/color/set1/655_1508459485.145504187.jpg",
                           "../resources/mission2/color/set1/655_1508459485.145504187.txt")
    };
*/

    ChangeDetector::DataVector queryData = {
            std::make_pair("../resources/mission2/color/set2/637_1508459484.2290233.jpg",
                           "../resources/mission2/color/set2/637_1508459484.2290233.txt"),
            std::make_pair("../resources/mission2/color/set2/634_1508459483.738083987.jpg",
                           "../resources/mission2/color/set2/634_1508459483.738083987.txt"),
            std::make_pair("../resources/mission2/color/set2/635_1508459483.796581763.jpg",
                           "../resources/mission2/color/set2/635_1508459483.796581763.txt"),
            std::make_pair("../resources/mission2/color/set2/636_1508459483.865743963.jpg",
                           "../resources/mission2/color/set2/636_1508459483.865743963.txt"),
            std::make_pair("../resources/mission2/color/set2/638_1508459484.45550608.jpg",
                           "../resources/mission2/color/set2/638_1508459484.45550608.txt"),
            std::make_pair("../resources/mission2/color/set2/639_1508459484.88277593.jpg",
                           "../resources/mission2/color/set2/639_1508459484.88277593.txt"),
            std::make_pair("../resources/mission2/color/set2/640_1508459484.145567010.jpg",
                           "../resources/mission2/color/set2/640_1508459484.145567010.txt")
    };

    ChangeDetector::DataVector referenceData = {
            std::make_pair("../resources/mission1/color/1135_1504729212.813591207.jpg",
                           "../resources/mission1/color/1135_1504729212.813591207.txt"),
            std::make_pair("../resources/mission1/color/1132_1504729212.613615958.jpg",
                           "../resources/mission1/color/1132_1504729212.613615958.txt"),
            std::make_pair("../resources/mission1/color/1133_1504729212.680179451.jpg",
                           "../resources/mission1/color/1133_1504729212.680179451.txt"),
            std::make_pair("../resources/mission1/color/1134_1504729212.746834314.jpg",
                           "../resources/mission1/color/1134_1504729212.746834314.txt"),
            std::make_pair("../resources/mission1/color/1136_1504729212.880206377.jpg",
                           "../resources/mission1/color/1136_1504729212.880206377.txt"),
            std::make_pair("../resources/mission1/color/1137_1504729212.946847205.jpg",
                           "../resources/mission1/color/1137_1504729212.946847205.txt"),
            std::make_pair("../resources/mission1/color/1138_1504729213.13646957.jpg",
                           "../resources/mission1/color/1138_1504729213.13646957.txt")
    };;

    changeDetection(queryData, referenceData);

    return 0;
}

void changeDetection(ChangeDetector::DataVector &queryImage, ChangeDetector::DataVector &trainImage) {
    auto start = std::chrono::high_resolution_clock::now();

    auto changeContext = new ChangeDetector::ChangeContextSURF(queryImage, trainImage);
    ChangeDetector::DetectChangeProcess process(changeContext);
    cv::Mat result = process.execute();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time (millisecods): " << duration << std::endl;

    showImage(result, "resulting differences");
}