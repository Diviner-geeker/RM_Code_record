#include "openvino_detect.h"
#define VIDEO_PATH "6radps.avi"
yolo_detct DEMO;
std::vector<yolo_detct::Object> result;
cv::TickMeter meter;
int main()
{
    cv::VideoCapture cap;
    cap.open(VIDEO_PATH);
    if (!cap.isOpened())
    {
        std::cout << "NO Videos" << std::endl;
        return 0;
    }
    while (true)
    {
        cv::Mat src_img;
        bool ret = cap.read(src_img);
        if (!ret)
            break;
        // meter.start();
        auto beforeTime = std::chrono::steady_clock::now();
        result = DEMO.work(src_img);
        for (auto i : result)
        {
            std::cout << "label:" << DEMO.class_names[i.label] << "  ";
            std::cout << "prob:" << i.prob << std::endl;
        }
        auto afterTime = std::chrono::steady_clock::now();
        double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
        std::cout << "fps:" << 1000.0 / duration_millsecond << std::endl;
        if (cv::waitKey(1) == 'q')
        {
            cap.release();
            std::cout << "finished by user\n";
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
