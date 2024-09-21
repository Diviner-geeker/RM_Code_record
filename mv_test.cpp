#include <opencv2/opencv.hpp>

#include "io/camera.hpp"

int main()
{
  io::Camera mvcamera(2);
  cv::Mat img;
  while (1) {
    mvcamera.read(img);
    if (img.empty()) {
      std::cout << "no img have been read" << std::endl;
      return 0;
    }
    cv::imshow("img", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }
  return 0;
}