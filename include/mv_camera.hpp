#ifndef IO__MV_CAMERA_HPP
#define IO__MV_CAMERA_HPP

#include <CameraApi.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <thread>

#include "thread_safe_queue.hpp"

using namespace std::chrono_literals;

namespace io
{
class MvCamera
{
public:
  MvCamera(double exposure_ms, double gamma, std::string serial)
  : exposure_ms_(exposure_ms),
    gamma_(gamma),
    _camera_id(serial),
    handle_(-1),
    quit_(false),
    ok_(false),
    queue_(1, []() { spdlog::debug("CameraData queue full!"); })
  {
    try_open();

    // 守护线程
    daemon_thread_ = std::thread{[this] {
      while (!quit_) {
        std::this_thread::sleep_for(100ms);

        if (ok_) continue;

        if (capture_thread_.joinable()) capture_thread_.join();

        close();
        try_open();
      }
    }};
  }

  MvCamera(double exposure_ms, double gamma)
  : exposure_ms_(exposure_ms),
    gamma_(gamma),
    handle_(-1),
    quit_(false),
    ok_(false),
    queue_(1, []() { spdlog::debug("CameraData queue full!"); })
  {
    try_open();

    // 守护线程
    daemon_thread_ = std::thread{[this] {
      while (!quit_) {
        std::this_thread::sleep_for(100ms);

        if (ok_) continue;

        if (capture_thread_.joinable()) capture_thread_.join();

        close();
        try_open();
      }
    }};
  }

  ~MvCamera()
  {
    quit_ = true;
    if (daemon_thread_.joinable()) daemon_thread_.join();
    if (capture_thread_.joinable()) capture_thread_.join();
    close();
    spdlog::info("Camera destructed.");
  }

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
  {
    CameraData data;
    queue_.pop(data);

    img = data.img;
    timestamp = data.timestamp;
  }

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  double exposure_ms_, gamma_;
  CameraHandle handle_;
  int height_, width_;
  bool quit_, ok_, _status;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  tools::ThreadSafeQueue<CameraData> queue_;
  std::string _camera_id;  //!< 指定相机的串口号

  void open()
  {
    int camera_num = 1;
    tSdkCameraDevInfo * camera_info_list;
    tSdkCameraCapbility camera_capbility;
    spdlog::info("open succeed");
    CameraSdkInit(1);
    spdlog::info("sdk init succeed");
    CameraEnumerateDevice(camera_info_list, &camera_num);
    spdlog::info("enum succeed");

    if (camera_num == 0) throw std::runtime_error("Not found camera!");

    // // Key: 序列号, Val: 相机设备信息哈希表
    // std::unordered_map<std::string, tSdkCameraDevInfo *> id_info;
    // for (INT enum_idx = 0; enum_idx < camera_num; ++enum_idx)
    //   id_info[camera_info_list[enum_idx].acSn] = &camera_info_list[enum_idx];

    // // spdlog::info("Serial number: {}", id_info);
    // for (auto it = id_info.begin(); it != id_info.end(); ++it) {
    //   std::cout << "Key: " << it->first << ", Val: " << it->second << std::endl;
    // }
    // for (auto it = id_info.begin(); it != id_info.end(); ++it) {
    //   delete it->second;
    // }
    // // 若无序列号，初始化第一个相机
    // if (_camera_id.empty()) _status = CameraInit(camera_info_list, -1, -1, &handle_);
    // // 根据序列号初始化相机
    // else if (id_info.find(_camera_id) != id_info.end())
    //   _status = CameraInit(id_info[_camera_id], -1, -1, &handle_);
    // // // 初始化失败
    // if (_status != CAMERA_STATUS_SUCCESS) throw std::runtime_error("Failed to init camera!");

    if (CameraInit(camera_info_list, -1, -1, &handle_) != CAMERA_STATUS_SUCCESS)
      throw std::runtime_error("Failed to init camera!");

    CameraGetCapability(handle_, &camera_capbility);
    width_ = camera_capbility.sResolutionRange.iWidthMax;
    height_ = camera_capbility.sResolutionRange.iHeightMax;

    CameraSetAeState(handle_, FALSE);                        // 关闭自动曝光
    CameraSetExposureTime(handle_, exposure_ms_ * 1e3);      // 设置曝光
    CameraSetGamma(handle_, gamma_ * 1e2);                   // 设置伽马
    CameraSetIspOutFormat(handle_, CAMERA_MEDIA_TYPE_BGR8);  // 设置输出格式为BGR
    CameraSetTriggerMode(handle_, 0);                        // 设置为连续采集模式
    CameraSetFrameSpeed(handle_, camera_capbility.iFrameSpeedDesc - 1);  // 以最高帧率运行

    CameraPlay(handle_);

    // 取图线程
    capture_thread_ = std::thread{[this] {
      tSdkFrameHead head;
      BYTE * raw;

      ok_ = true;
      while (!quit_) {
        auto img = cv::Mat(height_, width_, CV_8UC3);

        auto status = CameraGetImageBuffer(handle_, &head, &raw, 100);
        auto timestamp = std::chrono::steady_clock::now();

        if (status != CAMERA_STATUS_SUCCESS) {
          spdlog::warn("Camera dropped!");
          ok_ = false;
          break;
        }

        CameraImageProcess(handle_, raw, img.data, &head);
        CameraReleaseImageBuffer(handle_, raw);

        queue_.push({img, timestamp});
      }
    }};

    spdlog::info("Camera opened.");
  }

  void try_open()
  {
    try {
      open();
    } catch (const std::exception & e) {
      spdlog::warn("{}", e.what());
    }
  }

  void close()
  {
    if (handle_ == -1) return;
    CameraUnInit(handle_);
  }
};

}  // namespace io

#endif  // IO__MV_CAMERA_HPP