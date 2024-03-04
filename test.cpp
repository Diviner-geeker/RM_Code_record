#include <iostream>
#include <opencv2/opencv.hpp>

#include "include/hk_camera.hpp"
#include "include/mv_camera.hpp"

using namespace cv;
using namespace io;

int main()
{
  std::string hk_serial_01 = "00K64848135";
  std::string hk_serial_02 = "DA1351666";
  // std::string mv_serial_1 = "042071320239";
  // HkCamera hcamera01(6, 0.5, hk_serial_01);
  // HkCamera hcamera02(6, 0.5, hk_serial_02);
  std::cout << "start init" << std::endl;
  MvCamera mcamera(6, 0.5);
  std::cout << "init suceed" << std::endl;
  while (true) {
    cv::Mat img, img1;
    auto timestamp = std::chrono::steady_clock::now();
    // hcamera01.read(img, timestamp);
    // hcamera02.read(img1, timestamp);
    mcamera.read(img1, timestamp);
    // int depth = img.channels();
    // imshow("img", img);
    imshow("img1", img1);
    // std::cout << "depth:" << depth << std::endl;
    // std::cout << "01present frame rate" << hcamera01.pstValue.fCurValue << std::endl;
    // std::cout << "02present frame rate" << hcamera02.pstValue.fCurValue << std::endl;
    if (waitKey(1) == 'q') break;
  }

  return 0;
}