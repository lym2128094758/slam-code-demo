# apriltag_ros：AprilTag 检测

本目录包含 AprilTag 3 的 ROS 封装，支持相机流连续检测和单张图像检测，并发布检测结果及位姿。此版本的 `package.xml` 声明 BSD 许可证和上游仓库 [AprilRobotics/apriltag_ros](https://github.com/AprilRobotics/apriltag_ros)。

## 连续检测

默认相机话题是 `/camera_rect/image_rect` 与 `/camera_rect/camera_info`。例如 CRAIC 头部相机可这样启动：

```bash
roslaunch apriltag_ros continuous_detection.launch \
  camera_name:=/cam_h/color image_topic:=image_raw
```

这会订阅 `/cam_h/color/image_raw` 和 `/cam_h/color/camera_info`。检测配置从 `config/settings.yaml` 和 `config/tags.yaml` 加载；Tag ID、标签尺寸及 bundle 可在 `tags.yaml` 中查看和维护。检测结果话题由节点按上游实现发布，`yolo_grab.launch` 会将其接入 CRAIC 抓取控制器。

## 单张图像检测

```bash
roslaunch apriltag_ros single_image_server.launch
```

客户端 launch 需要指定输入、输出图像路径：

```bash
roslaunch apriltag_ros single_image_client.launch \
  image_load_path:=/path/to/input.png image_save_path:=/path/to/output.png
```

单张图像模式使用 launch 文件里的相机内参；用于真实相机时应按相机标定结果修改内参。Tag 尺寸以米为单位，位姿精度依赖准确的相机内参与标签实际尺寸。

## 目录

- `src/`、`include/`：连续检测、单图检测和公共 C++ 实现。
- `msg/`、`srv/`：检测消息与单图分析服务定义。
- `config/`：检测参数和 AprilTag ID/尺寸配置。
- `docs/tag_size_guide.svg`：标签尺寸测量说明。
