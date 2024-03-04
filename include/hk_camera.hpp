#ifndef IO__HK_CAMERA_HPP
#define IO__HK_CAMERA_HPP

#include <spdlog/spdlog.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "MvCameraControl.h"
#include "thread_safe_queue.hpp"

using namespace std::chrono_literals;

namespace io
{
class HkCamera
{
public:
  MVCC_FLOATVALUE pstValue;
  std::string serial_;
  HkCamera(double exposure_ms, double gamma, std::string_view serial)
  : exposure_ms_(exposure_ms),
    gamma_(gamma),
    serial_(serial),
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

  ~HkCamera()
  {
    quit_ = true;
    if (daemon_thread_.joinable()) daemon_thread_.join();
    if (capture_thread_.joinable()) capture_thread_.join();
    close();
    spdlog::info("Camera destructed.");
  }

  cv::ColorConversionCodes pixelType2CVType(MvGvspPixelType pixel_type)
  {
    static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> convertion = {
      {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
      {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
      {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
      {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
    return convertion[pixel_type];
  }

  bool retrieve(cv::OutputArray image)
  {
    auto & frame_info = _p_out.stFrameInfo;
    // 当前格式
    auto pixel_type = frame_info.enPixelType;
    // 单通道标志位集合
    std::unordered_set<MvGvspPixelType> mono_set = {
      PixelType_Gvsp_Mono1p,        PixelType_Gvsp_Mono2p,       PixelType_Gvsp_Mono4p,
      PixelType_Gvsp_Mono8,         PixelType_Gvsp_Mono8_Signed, PixelType_Gvsp_Mono10,
      PixelType_Gvsp_Mono10_Packed, PixelType_Gvsp_Mono12,       PixelType_Gvsp_Mono12_Packed,
      PixelType_Gvsp_Mono14,        PixelType_Gvsp_Mono16};
    if (pixel_type == PixelType_Gvsp_Undefined) {
      std::cout << "hik - undefined pixel type" << std::endl;
      image.assign(cv::Mat());
      return false;
    }
    // 通道数
    uint channel = mono_set.find(pixel_type) == mono_set.end() ? 3 : 1;
    auto frame_size = frame_info.nWidth * frame_info.nHeight * channel;
    if (_p_dstbuf.size() != frame_size) _p_dstbuf.resize(frame_size);
    // cv::cvtColor
    cv::Mat src_image(cv::Size(frame_info.nWidth, frame_info.nHeight), CV_8U, _p_out.pBufAddr);
    if (channel == 1)
      image.assign(src_image);
    else {
      cv::Mat dst_image;
      cv::cvtColor(src_image, dst_image, pixelType2CVType(pixel_type));
      image.assign(dst_image);
    }
    return true;
  }

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
  {
    CameraData data;
    queue_.pop(data);

    img = data.img;
    timestamp = data.timestamp;
  }

  void open()
  {
    MV_CC_DEVICE_INFO_LIST camera_info_list;
    //初始化SDK
    MV_CC_Initialize();
    //枚举设备
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &camera_info_list);
    unsigned int camera_num = camera_info_list.nDeviceNum;
    spdlog::info("Camera num: {}", camera_num);
    if (camera_num == 0) throw std::runtime_error("Not found camera!");

    std::unordered_map<std::string, MV_CC_DEVICE_INFO *> sn_device;
    // 设备列表指针数组
    auto device_info_arr = camera_info_list.pDeviceInfo;
    // 哈希匹配
    for (decltype(camera_num) i = 0; i < camera_num; ++i) {
      if (device_info_arr[i] == nullptr) break;
      std::string sn =
        reinterpret_cast<char *>(device_info_arr[i]->SpecialInfo.stUsb3VInfo.chSerialNumber);
      spdlog::info("Serial number: {}", sn);
      sn_device[sn] = device_info_arr[i];
    }
    // 查找指定设备，否则选取第一个设备
    if (sn_device.find(serial_) != sn_device.end())
      nRet = MV_CC_CreateHandle(&handle_, sn_device[serial_]);
    else
      nRet = MV_CC_CreateHandle(&handle_, device_info_arr[0]);
    if (nRet != MV_OK) throw std::runtime_error("Fail to create handle!");

    // MV_CC_CreateHandle(&handle_, camera_info_list.pDeviceInfo[0]);

    if (MV_CC_OpenDevice(handle_) != MV_OK) throw std::runtime_error("Failed to open camera!");
    MV_CC_GetFrameRate(handle_, &pstValue);
    MV_CC_SetExposureAutoMode(handle_, 0);               // 关闭自动曝光
    MV_CC_SetExposureTime(handle_, exposure_ms_ * 1e3);  // 设置曝光
    MV_CC_SetGamma(handle_, gamma_ * 1e2);               // 设置伽马
    // MV_CC_SetPixelFormat(handle_, PixelType_Gvsp_BGR8_Packed);            // 设置输出格式为BGR
    MV_CC_SetEnumValue(handle_, "TriggerMode", 0);  // 设置为连续采集模式
    MV_CC_SetFrameRate(handle_, pstValue.fMax);     // 设置特定帧率运行
    MV_CC_StartGrabbing(handle_);                   //开始取流
    std::cout << "max frame rate:" << pstValue.fMax << std::endl;

    // 取图线程
    cv::Mat img;
    capture_thread_ = std::thread{[this, img] {
      ok_ = true;
      while (!quit_) {
        // 获取图像地址
        auto ret = MV_CC_GetImageBuffer(handle_, &_p_out, 1000);
        auto timestamp = std::chrono::steady_clock::now();
        if (ret == MV_OK)
          retrieve(img);
        else {
          std::cout << "No data in getting image buffer" << std::endl;
        }
        // 释放图像缓存
        if (_p_out.pBufAddr != nullptr) {
          ret = MV_CC_FreeImageBuffer(handle_, &_p_out);
          if (ret != MV_OK) {
            std::cout << "hik - failed to free image buffer " << std::endl;
          }
        }
        queue_.push({img, timestamp});
        // spdlog::info("queue push success");
      }
    }};
    spdlog::info("Camera opened.");
  }

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };
  double exposure_ms_, gamma_;
  MV_FRAME_OUT _p_out;
  void * handle_;
  bool quit_, ok_;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  std::vector<uchar> _p_dstbuf;  //!< 输出数据缓存
  tools::ThreadSafeQueue<CameraData> queue_;

  int nRet = MV_OK;

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
    MV_CC_StopGrabbing(handle_);
    MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    MV_CC_Finalize();
  }
};

}  // namespace io

#endif  // IO__MV_CAMERA_HPP