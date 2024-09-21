#include <opencv2/opencv.hpp>

#include "io/camera.hpp"

int main()
{
  io::Camera mvcamera(2);
  cv::Mat img;
  mvcamera.read(img);
  cv::imshow("img", img);
  cv::waitKey(0);
  return 0;
}