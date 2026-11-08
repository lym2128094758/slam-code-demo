# yolov8_detection：视觉目标与零件检测

本包将 YOLOv8 检测接入 ROS 图像话题，提供普通目标检测、零件分类和坐标发布脚本。核心依赖包括 `rospy`、`cv_bridge`、OpenCV、NumPy、PyTorch 与 `ultralytics`；模型推理必须使用与 ROS Python 环境兼容的依赖版本。

## 常用启动方式

```bash
# 黑白零件检测，使用仓库中的自训练模型 best2.pt
roslaunch yolov8_detection yolo_parts.launch

# AprilTag 检测、YOLO目标检测和 CRAIC 抓取控制器的组合入口
roslaunch yolov8_detection yolo_grab.launch

# 在目标 Tag ID 为 7 时尝试自动抓取
roslaunch yolov8_detection yolo_grab.launch auto:=true target_tag:=7
```

`yolo_parts.launch` 默认加载 `scripts/models/best2.pt`，订阅 `/cam_h/color/image_raw`、`/cam_h/depth/image_raw/compressedDepth` 和 `/cam_h/color/camera_info`，将零件位置发布到 `/yolo_parts/detections`。`yolo_grab.launch` 默认使用 `/cam_h/color/image_raw` 和 `/cam_h/color/camera_info`，并包含 `apriltag_ros` 连续检测及 `craic_task_template/grab_controller.py`；`auto` 默认关闭。

## 脚本说明

| 脚本 | 用途 |
| --- | --- |
| `yolo_detector.py` | 通用目标检测，默认监听 `/camera/...` 话题，默认模型参数为 `yolov8n.pt` |
| `yolo_parts.py` | 黑白零件检测与深度坐标转换，默认使用头部相机话题 |
| `hand_camera_yolo_detector.py` | 手部相机检测脚本，可通过 ROS 参数设置相机命名空间和模型 |
| `yolo_listener.py` | 读取 Kuavo 自定义检测消息的监听示例 |
| 文件名带“复件”或说明性中文后缀的脚本 | 历史试验或变体，供查阅；不是上述两个常用 launch 的默认入口 |

模型文件 `best.pt`、`best2.pt` 与 `best（）2.pt` 均随资料保留。`best2.pt` 和 `best（）2.pt` 内容相同；保留原始文件名，不影响 launch 使用 `best2.pt`。

## 环境要求

在 ROS Noetic 工作区编译本包后，确认仿真相机正在发布 launch 中所需的图像、深度和相机内参话题。安装 `ultralytics` 时请匹配已安装的 PyTorch/CUDA 版本；在没有图形桌面的环境中，可通过相应节点的 `show_window` 参数关闭图像窗口。

组合抓取入口还依赖 `apriltag_ros`、`craic_task_template`、Kuavo/CRAIC 消息类型与仿真夹爪控制模块。它们不由本包独立提供。
