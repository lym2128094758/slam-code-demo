#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手部相机 YOLOv8 目标检测器 - 专门用于视觉追踪IK
不做TF变换到base_link，只发布相机坐标系下的位置

订阅:
    /cam_r/color/image_raw           — 右手RGB图像
    /cam_r/aligned_depth_to_color/image_raw — 右手深度图像
    /cam_r/color/camera_info         — 右手相机内参

发布:
    /yolo_hand_target                — 手部相机检测目标（相机坐标系）
"""

import sys
import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

try:
    from ultralytics import YOLO
except ImportError:
    rospy.logerr("❌ 未安装 ultralytics, 请执行: pip install ultralytics")
    sys.exit(1)


# ============================ 常量 / 配置 ============================

# 深度采样参数
DEPTH_SAMPLE_REGIONS = 9
DEPTH_MEDIAN_BLUR_KERNEL = 5
DEPTH_CONSISTENCY_THRESH = 0.2

# YOLO 配置
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7

# 深度范围过滤
MIN_DEPTH = 0.1
MAX_DEPTH = 2.0


class HandCameraYoloDetector:
    def __init__(self):
        rospy.init_node('hand_camera_yolo_detector', anonymous=True)

        model_path = rospy.get_param('~model', 'yolov8n.pt')
        self.conf_threshold = rospy.get_param('~conf_thresh', CONF_THRESHOLD)
        self.use_left = rospy.get_param('~use_left', False)

        self.model = YOLO(model_path)
        rospy.loginfo(f"✅ 手部相机YOLO模型加载成功: {model_path}")

        self.bridge = CvBridge()

        self.camera_info = None
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0

        self.depth_image = None
        self.depth_timestamp = rospy.Time(0)
        self.fallback_depth = rospy.get_param('~fallback_depth', 0.3)  # 无深度时的备用距离(m)

        self.hand_target_pub = rospy.Publisher(
            '/yolo_hand_target', PointStamped, queue_size=10)

        if self.use_left:
            rospy.loginfo("🤚 使用左手相机 (cam_l)")
            cam_ns = '/cam_l/color'
            depth_candidates = [
                '/cam_l/aligned_depth_to_color/image_raw',
                '/cam_l/depth/image_raw',
                '/cam_l/depth/image_rect_raw',
            ]
        else:
            rospy.loginfo("✋ 使用右手相机 (cam_r)")
            cam_ns = '/cam_r/color'
            depth_candidates = [
                '/cam_r/aligned_depth_to_color/image_raw',
                '/cam_r/depth/image_raw',
                '/cam_r/depth/image_rect_raw',
            ]

        rospy.Subscriber(f'{cam_ns}/image_raw', Image,
                         self.image_callback, queue_size=1)
        rospy.Subscriber(f'{cam_ns}/camera_info', CameraInfo,
                         self.camera_info_callback, queue_size=1)

        # 先尝试订阅原始深度话题，同时也订阅压缩深度话题
        for depth_topic in depth_candidates:
            rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
            rospy.loginfo(f"📡 订阅深度话题: {depth_topic}")

        # 订阅压缩深度话题（cam_r 实际只发布这个）
        if self.use_left:
            compressed_depth_topic = '/cam_l/depth/image_rect_raw/compressedDepth'
        else:
            compressed_depth_topic = '/cam_r/depth/image_rect_raw/compressedDepth'
        rospy.Subscriber(compressed_depth_topic, CompressedImage,
                         self.compressed_depth_callback, queue_size=1)
        rospy.loginfo(f"📡 订阅压缩深度话题: {compressed_depth_topic}")

        rospy.loginfo(f"🎯 手部相机YOLOv8 Detector 启动 (fallback_depth={self.fallback_depth}m)")

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        # rospy.loginfo(f"📷 手部相机内参: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_timestamp = msg.header.stamp
            rospy.loginfo_throttle(10.0, f"📊 手部深度图像来自: {msg.header.frame_id} topic接收正常")
        except Exception as e:
            rospy.logwarn(f"手部深度图像转换失败: {e}")

    def compressed_depth_callback(self, msg):
        """解码 compressedDepth 压缩深度图像（16UC1 PNG格式）"""
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            # compressedDepth 可能有12字节配置头，先尝试直接解码
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None and len(buf) > 12:
                # 跳过12字节头部再试
                img = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.depth_image = img
                self.depth_timestamp = msg.header.stamp
                rospy.loginfo_throttle(10.0, "📊 压缩深度图像解码成功")
            else:
                rospy.logwarn_throttle(5.0, "⚠️  压缩深度图像解码失败")
        except Exception as e:
            rospy.logwarn(f"压缩深度解码异常: {e}")

    def _get_depth_patch(self, u, v, w, h, num_regions=DEPTH_SAMPLE_REGIONS):
        if self.depth_image is None:
            return []

        grid_size = int(np.sqrt(num_regions))
        step_x = w / (grid_size + 1)
        step_y = h / (grid_size + 1)

        valid_depths = []
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                sample_u = int(u + i * step_x)
                sample_v = int(v + j * step_y)
                if 0 <= sample_v < self.depth_image.shape[0] and \
                   0 <= sample_u < self.depth_image.shape[1]:
                    depth_mm = self.depth_image[sample_v, sample_u]
                    depth_m = depth_mm / 1000.0
                    if MIN_DEPTH <= depth_m <= MAX_DEPTH:
                        valid_depths.append(depth_m)
        return valid_depths

    def _estimate_depth(self, u, v, w, h):
        valid_depths = self._get_depth_patch(u, v, w, h)
        if not valid_depths:
            rospy.logwarn_throttle(2.0, f"⚠️  无有效深度采样 (depth_image={'存在' if self.depth_image is not None else '不存在'})")
            return None

        filtered_depths = cv2.medianBlur(
            np.array(valid_depths, dtype=np.float32).reshape(1, -1),
            DEPTH_MEDIAN_BLUR_KERNEL
        ).flatten()

        median_depth = np.median(filtered_depths)
        mean_depth = np.mean(filtered_depths)
        final_depth = median_depth * 0.8 + mean_depth * 0.2

        depth_std = np.std(filtered_depths)
        depth_consistency = 1.0 - min(depth_std / mean_depth, 1.0)
        
        if depth_consistency < 0.5:
            rospy.logwarn(f"⚠️  手部深度一致性低 ({depth_consistency:.2f})")

        return final_depth

    def _pixel_to_camera(self, u, v, depth):
        if self.fx == 0 or self.fy == 0:
            return None
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return (x, y, z)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.model(frame_rgb, verbose=False,
                                conf=self.conf_threshold, iou=IOU_THRESHOLD)

            best_result = None
            best_conf = 0.0
            detected_names = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cls]
                    detected_names.append(f"{name}:{conf:.2f}")

                    if conf > best_conf:
                        best_conf = conf
                        best_result = {
                            'x1': int(box.xyxy[0][0]),
                            'y1': int(box.xyxy[0][1]),
                            'x2': int(box.xyxy[0][2]),
                            'y2': int(box.xyxy[0][3]),
                            'conf': conf,
                            'name': name,
                        }

            if detected_names:
                pass  # rospy.loginfo_throttle(2.0, f"🔍 手部YOLO检测到: {detected_names}")

            if best_result:
                u = (best_result['x1'] + best_result['x2']) / 2
                v = (best_result['y1'] + best_result['y2']) / 2
                w = best_result['x2'] - best_result['x1']
                h = best_result['y2'] - best_result['y1']

                # rospy.loginfo_throttle(2.0, f"🤏 手部检测: {best_result['name']} 置信度={best_conf:.2f}")

                depth = self._estimate_depth(u, v, w, h)
                if depth is None:
                    # 深度不可用时使用固定fallback深度，仍然发布x,y供视觉追踪使用
                    depth = self.fallback_depth
                    rospy.logwarn_throttle(3.0,
                        f"⚠️  深度不可用，使用fallback_depth={depth:.2f}m")

                camera_pt = self._pixel_to_camera(u, v, depth)
                if camera_pt is None:
                    return

                cx, cy, cz = camera_pt

                point_msg = PointStamped()
                point_msg.header = Header()
                point_msg.header.stamp = rospy.Time.now()
                point_msg.header.frame_id = msg.header.frame_id
                point_msg.point.x = cx
                point_msg.point.y = cy
                point_msg.point.z = cz

                self.hand_target_pub.publish(point_msg)
                rospy.loginfo_throttle(
                    1.0,
                    f"📍 手部目标发布: x={cx:.3f} y={cy:.3f} z={cz:.3f}")

        except Exception as e:
            rospy.logerr(f"手部图像处理异常: {e}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        HandCameraYoloDetector().run()
    except rospy.ROSInterruptException:
        rospy.loginfo("手部相机YOLO检测器已停止")
