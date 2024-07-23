#include "yolov8.hpp"

#include <omp.h>

#include <algorithm>
#include <random>

namespace tools
{
std::vector<cv::Scalar> class_colors;

auto net = cv::dnn::readNetFromONNX("model/tiny_resnet.onnx");

cv::Scalar get_color(int class_id)
{
  while (class_id >= class_colors.size()) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    class_colors.emplace_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
  }
  return class_colors[class_id];
}

void draw_detections(
  cv::Mat & img, const std::vector<Detection> & detections,
  const std::vector<std::string> & classes)
{
  //change
  cv::Scalar kp_color = cv::Scalar(0, 255, 0);
  int radius = 3;
  //
  for (const auto & d : detections) {
    auto box = d.box;
    // auto color = get_color(d.class_id);
    auto label = (classes.empty()) ? std::to_string(d.class_id) : classes[d.class_id];
    auto text = label + " " + ARMOR_NAMES[d.name] + std::to_string(d.confidence).substr(0, 4);
    // auto text_size = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
    // auto text_box = cv::Rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
    //change
    for (const auto & point : d.armor_keypoints) {
      cv::circle(img, point, radius, kp_color);
    }
    for (int j = 0; j < 4; j++) {
      cv::line(img, d.armor_keypoints[j], d.armor_keypoints[(j + 1) % 4], kp_color, 1);
    }
    //
    // cv::rectangle(img, box, color, 2);
    // cv::rectangle(img, text_box, color, cv::FILLED);
    cv::circle(img, d.center, radius, cv::Scalar(255, 0, 255));
    cv::putText(img, text, {box.x, box.y}, cv::FONT_HERSHEY_DUPLEX, 0.8, kp_color);
  }
}

static void sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = bottom_points[0];  // bottom-left
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = top_points[1];     // top-right
}

cv::Mat YOLOV8::get_pattern(const cv::Mat & bgr_img, const Detection & detection) const
{
  // 延长灯条获得装甲板角点
  // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
  auto tl = (detection.armor_keypoints[0] + detection.armor_keypoints[1]) / 2 -
            (detection.armor_keypoints[1] - detection.armor_keypoints[0]) * 1.125;
  auto bl = (detection.armor_keypoints[0] + detection.armor_keypoints[1]) / 2 +
            (detection.armor_keypoints[1] - detection.armor_keypoints[0]) * 1.125;
  auto tr = (detection.armor_keypoints[2] + detection.armor_keypoints[3]) / 2 -
            (detection.armor_keypoints[2] - detection.armor_keypoints[3]) * 1.125;
  auto br = (detection.armor_keypoints[2] + detection.armor_keypoints[3]) / 2 +
            (detection.armor_keypoints[2] - detection.armor_keypoints[3]) * 1.125;
  // std::cout << "left_top:" << tl << "  "
  //           << "left_bottom:" << bl << std::endl;
  // std::cout << "right_top:" << tr << "  "
  //           << "right_bottom:" << br << std::endl;
  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  auto roi_tl = cv::Point(roi_left, roi_top);
  auto roi_br = cv::Point(roi_right, roi_bottom);
  auto roi = cv::Rect(roi_tl, roi_br);
  return bgr_img(roi);
}

static void classify(Detection & armor)
{
  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});

  auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

  net.setInput(blob);
  cv::Mat outputs = net.forward();

  // softmax
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

YOLOV8::YOLOV8(const std::string & model_path, int class_num, const std::string & device)
: class_num_(class_num)
{
  auto model = core_.read_model(model_path);

  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 416, 416, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

std::vector<Detection> YOLOV8::infer(const cv::Mat & bgr_img)
{
  auto x_scale = static_cast<double>(416) / bgr_img.rows;
  auto y_scale = static_cast<double>(416) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // preproces
  auto input = cv::Mat(416, 416, CV_8UC3);
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});
  ov::Tensor input_tensor(ov::element::u8, {1, 416, 416, 3}, input.data);

  /// infer
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, bgr_img);
}

std::vector<Detection> YOLOV8::parse(double scale, cv::Mat & output, const cv::Mat & bgr_img) const
{
  // for each row: xywh + classess
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  //change here
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  //

  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    //change here
    auto one_key_points = output.row(r).colRange(4 + class_num_, 14);
    std::vector<cv::Point2f> armor_key_points;
    //

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) continue;

    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);
    //change here
    for (int i = 0; i < 4; i++) {
      float x = one_key_points.at<float>(0, i * 2 + 0) / scale;
      float y = one_key_points.at<float>(0, i * 2 + 1) / scale;
      cv::Point2f kp = {x, y};
      armor_key_points.push_back(kp);
    }
    //

    ids.emplace_back(max_point.x);
    confidences.emplace_back(score);
    boxes.emplace_back(left, top, width, height);
    //change here
    armors_key_points.emplace_back(armor_key_points);
    //
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::vector<Detection> detections;
  for (const auto & i : indices) {
    //change here
    sort_keypoints(armors_key_points[i]);

    detections.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    //
  }
  for (auto & d : detections) {
    d.pattern = get_pattern(bgr_img, d);
    classify(d);
    // cv::imshow("pattern", d.pattern);
  }
  return detections;
}

}  // namespace tools
