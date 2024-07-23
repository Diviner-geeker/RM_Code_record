#ifndef TOOLS__YOLOV8_HPP
#define TOOLS__YOLOV8_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace tools
{

enum ArmorName { one, two, three, four, five, sentry, outpost, base, not_armor };
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};
struct Detection
{
  int class_id;
  float confidence;
  ArmorName name;
  cv::Rect box;
  cv::Mat pattern;
  cv::Point2f center;
  //change here
  std::vector<cv::Point2f> armor_keypoints;

  Detection(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints)
  : class_id(class_id), confidence(confidence), box(box), armor_keypoints(armor_keypoints)
  {
    center =
      (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  }
};

void draw_detections(
  cv::Mat & img, const std::vector<Detection> & detections,
  const std::vector<std::string> & classes = {});

class YOLOV8
{
public:
  YOLOV8(const std::string & model_path, int class_num, const std::string & device = "AUTO");

  std::vector<Detection> infer(const cv::Mat & bgr_img);

private:
  int class_num_;
  float nms_threshold_ = 0.3;
  float score_threshold_ = 0.7;
  ov::Core core_;
  ov::CompiledModel compiled_model_;

  std::vector<Detection> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img) const;
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Detection & detection) const;
};

}  // namespace tools

#endif  // TOOLS__YOLOV8_HPP