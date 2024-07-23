#include <chrono>
#include <opencv2/opencv.hpp>

#include "yolov8.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{@video_path    |                        | avi路径}";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto video_path = cli.get<std::string>(0);
  // std::vector<std::string> classes = {"armor"};
  std::vector<std::string> classes = {"blue", "red"};
  cv::VideoCapture video(video_path);

  tools::YOLOV8 yolo("model/last.xml", classes.size(), "AUTO");

  auto sum = 0.0;
  auto count = 0;

  bool pause = false;

  while (true) {
    cv::Mat img;
    video.read(img);
    if (img.empty()) break;
    if (pause) {
      auto w = cv::waitKey(0);
      if (w == ' ') {
        pause = !pause;
      }
    }

    auto start = std::chrono::steady_clock::now();

    auto detections = yolo.infer(img);

    auto end = std::chrono::steady_clock::now();
    auto span = std::chrono::duration<double>(end - start).count();
    std::cout << "total: " << span * 1e3 << "ms\n";

    sum += span;
    count += 1;

    tools::draw_detections(img, detections, classes);

    cv::imshow("press q to quit", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
    if (key == ' ') pause = true;
  }

  std::cout << "avg: " << count / sum << "fps\n";

  return 0;
}