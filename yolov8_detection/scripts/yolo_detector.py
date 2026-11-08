#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 目标检测器 v3 — 优化深度估计 + TF坐标变换

改进要点：
1. 深度估计优化：多区域采样 + 中值滤波 + 深度一致性检查
2. TF坐标变换：自动匹配相机坐标系，发布 base_link 坐标
3. 发布最佳目标到 /yolo_best_target 话题

订阅:
    /camera/color/image_raw   — RGB图像
    /camera/aligned_depth_to_color/image_raw — 深度图像

发布:
    /yolo_detection        — 检测结果（含像素坐标）
    /yolo_target_info      — 目标详细信息（含3D坐标）
    /yolo_best_target      — 最佳抓取目标（PointStamped in base_link）
"""

import sys
import os
import cv2
import numpy as np
import rospy
import tf
import tf.transformations as tr
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

try:
    from ultralytics import YOLO
except ImportError:
    rospy.logerr("❌ 未安装 ultralytics, 请执行: pip install ultralytics")
    sys.exit(1)


# ============================ 常量 / 配置 ============================

# 深度采样参数
DEPTH_SAMPLE_REGIONS = 9       # 采样区域数量（3x3网格）
DEPTH_MEDIAN_BLUR_KERNEL = 5   # 中值滤波核大小
DEPTH_CONSISTENCY_THRESH = 0.2 # 深度一致性阈值（标准差/均值）

# YOLO 配置
CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.7
TARGET_CLASS_NAME = "bottle"   # 目标类别

# TF 变换配置
CAMERA_FRAMES = [
    'camera_color_optical_frame',   # Realsense 默认
    'Head Camera View',             # MuJoCo Kuavo 52
    'cam_h_color_optical_frame',    # 仿真相机
    'camera_depth_frame',           # 深度相机
]

# 深度范围过滤
MIN_DEPTH = 0.1   # 最小有效深度 (m)
MAX_DEPTH = 2.0   # 最大有效深度 (m)


# ============================ YOLO 检测器 ============================

class YoloDetector:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)

        # ---- 参数 ----
        model_path = rospy.get_param('~model', 'yolov8n.pt')
        self.target_class = rospy.get_param('~target_class', TARGET_CLASS_NAME)
        self.conf_threshold = rospy.get_param('~conf_thresh', CONF_THRESHOLD)

        # ---- 初始化 YOLO 模型 ----
        self.model = YOLO(model_path)
        rospy.loginfo(f"✅ YOLO 模型加载成功: {model_path}")

        # ---- CV Bridge ----
        self.bridge = CvBridge()

        # ---- TF 监听器 ----
        self.tf_listener = tf.TransformListener()

        # ---- 相机内参缓存 ----
        self.camera_info = None
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0

        # ---- 深度图像缓存 ----
        self.depth_image = None
        self.depth_timestamp = rospy.Time(0)

        # ---- 发布者 ----
        self.best_target_pub = rospy.Publisher(
            '/yolo_best_target', PointStamped, queue_size=10)

        # ---- 订阅者 ----
        rospy.Subscriber('/camera/color/image_raw', Image,
                         self.image_callback, queue_size=1)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,
                         self.depth_callback, queue_size=1)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo,
                         self.camera_info_callback, queue_size=1)

        rospy.loginfo("🎯 YOLOv8 Detector v3 启动")

    def camera_info_callback(self, msg):
        """获取相机内参"""
        self.camera_info = msg
        self.fx = msg.K[0]  # fx
        self.fy = msg.K[4]  # fy
        self.cx = msg.K[2]  # cx
        self.cy = msg.K[5]  # cy
        rospy.loginfo(f"📷 相机内参: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def depth_callback(self, msg):
        """缓存深度图像"""
        try:
            # 深度图像格式为 16-bit uint，单位为毫米
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_timestamp = msg.header.stamp
        except Exception as e:
            rospy.logwarn(f"深度图像转换失败: {e}")

    def _get_depth_patch(self, u, v, w, h, num_regions=DEPTH_SAMPLE_REGIONS):
        """
        在检测框内进行多区域采样获取深度值
        返回: 有效深度值列表（单位：米）
        """
        if self.depth_image is None:
            rospy.logwarn("⚠️  无深度图像")
            return []

        # 计算采样网格
        grid_size = int(np.sqrt(num_regions))
        step_x = w / (grid_size + 1)
        step_y = h / (grid_size + 1)

        valid_depths = []
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                sample_u = int(u + i * step_x)
                sample_v = int(v + j * step_y)

                # 边界检查
                if 0 <= sample_v < self.depth_image.shape[0] and \
                   0 <= sample_u < self.depth_image.shape[1]:
                    depth_mm = self.depth_image[sample_v, sample_u]
                    depth_m = depth_mm / 1000.0  # 转换为米
                    if MIN_DEPTH <= depth_m <= MAX_DEPTH:
                        valid_depths.append(depth_m)

        return valid_depths

    def _estimate_depth(self, u, v, w, h):
        """
        优化的深度估计方法：
        1. 多区域采样获取深度补丁
        2. 中值滤波去除噪声
        3. 深度一致性检查
        """
        # 获取采样深度
        valid_depths = self._get_depth_patch(u, v, w, h)
        if not valid_depths:
            rospy.logwarn("⚠️  无有效深度采样")
            return None

        # 中值滤波
        filtered_depths = cv2.medianBlur(
            np.array(valid_depths, dtype=np.float32).reshape(1, -1),
            DEPTH_MEDIAN_BLUR_KERNEL
        ).flatten()

        # 中位数+均值融合
        median_depth = np.median(filtered_depths)
        mean_depth = np.mean(filtered_depths)
        final_depth = median_depth * 0.8 + mean_depth * 0.2

        # 深度一致性检查
        depth_std = np.std(filtered_depths)
        depth_consistency = 1.0 - min(depth_std / mean_depth, 1.0)
        
        if depth_consistency < 0.5:
            rospy.logwarn(f"⚠️  深度一致性低 ({depth_consistency:.2f}), 深度可能不可靠")

        rospy.loginfo_throttle(
            2.0,
            f"� 深度估计: 采样数={len(valid_depths)}, "
            f"中值={median_depth:.3f}m, 均值={mean_depth:.3f}m, "
            f"融合={final_depth:.3f}m, 一致性={depth_consistency:.2f}"
        )

        return final_depth

    def _pixel_to_camera(self, u, v, depth):
        """
        将像素坐标转换为相机坐标系下的3D坐标
        使用针孔相机模型
        """
        if self.fx == 0 or self.fy == 0:
            rospy.logwarn("⚠️  相机内参未初始化")
            return None

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return (x, y, z)

    def _transform_to_base_link(self, x, y, z, source_frame):
        """
        将相机坐标系下的点变换到 base_link
        自动尝试多个可能的相机 frame
        """
        frames = []
        if source_frame:
            frames.append(source_frame)
        for f in CAMERA_FRAMES:
            if f not in frames:
                frames.append(f)

        for frame in frames:
            try:
                (trans, rot) = self.tf_listener.lookupTransform(
                    'base_link', frame, rospy.Time(0))
                T = tr.quaternion_matrix(rot)
                T[0:3, 3] = trans
                pt = np.dot(T, [x, y, z, 1.0])
                rospy.loginfo(f"✅ TF 变换成功: {frame} → base_link")
                return pt[0], pt[1], pt[2]
            except Exception as e:
                continue

        rospy.logwarn(f"⚠️  无法找到 TF 变换: {source_frame} → base_link")
        return None

    def image_callback(self, msg):
        """处理 RGB 图像，执行检测"""
        try:
            # 转换 ROS 图像到 OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 执行 YOLO 检测
            results = self.model(frame_rgb, verbose=False,
                                conf=self.conf_threshold, iou=IOU_THRESHOLD)

            # 获取最佳目标（置信度最高的目标）
            best_result = None
            best_conf = 0.0
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cls]

                    if name == self.target_class and conf > best_conf:
                        best_conf = conf
                        best_result = {
                            'x1': int(box.xyxy[0][0]),
                            'y1': int(box.xyxy[0][1]),
                            'x2': int(box.xyxy[0][2]),
                            'y2': int(box.xyxy[0][3]),
                            'conf': conf,
                            'name': name,
                        }

            if best_result:
                # 计算检测框中心
                u = (best_result['x1'] + best_result['x2']) / 2
                v = (best_result['y1'] + best_result['y2']) / 2
                w = best_result['x2'] - best_result['x1']
                h = best_result['y2'] - best_result['y1']

                rospy.loginfo(
                    f"🎯 检测到目标: {best_result['name']} "
                    f"置信度={best_conf:.2f} "
                    f"位置=({u:.0f},{v:.0f}) "
                    f"尺寸={w:.0f}x{h:.0f}")

                # 深度估计
                depth = self._estimate_depth(u, v, w, h)
                if depth is None:
                    rospy.logwarn("⚠️  无法获取深度，跳过发布")
                    return

                # 像素坐标转相机坐标
                camera_pt = self._pixel_to_camera(u, v, depth)
                if camera_pt is None:
                    return

                # 相机坐标转 base_link
                base_link_pt = self._transform_to_base_link(
                    camera_pt[0], camera_pt[1], camera_pt[2],
                    msg.header.frame_id)
                
                if base_link_pt is None:
                    return

                bx, by, bz = base_link_pt

                # 发布最佳目标（base_link 坐标系）
                point_msg = PointStamped()
                point_msg.header = Header()
                point_msg.header.stamp = rospy.Time.now()
                point_msg.header.frame_id = 'base_link'
                point_msg.point.x = bx
                point_msg.point.y = by
                point_msg.point.z = bz

                self.best_target_pub.publish(point_msg)
                rospy.loginfo(
                    f"📍 发布目标 (base_link): x={bx:.3f} y={by:.3f} z={bz:.3f}")

        except Exception as e:
            rospy.logerr(f"图像处理异常: {e}")
            import traceback; traceback.print_exc()

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        YoloDetector().run()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLO 检测器已停止")
