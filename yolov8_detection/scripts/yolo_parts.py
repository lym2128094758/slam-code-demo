#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 黑白零件检测器 - 实时显示检测结果与置信度

订阅:
    /cam_h/color/image_raw — 头部相机RGB图像
    /cam_h/depth/image_raw/compressedDepth — 深度图像（压缩格式）
    /cam_h/color/camera_info — 相机内参

发布:
    /yolo_parts/detections — 检测结果（base_link坐标系）

显示:
    实时窗口显示检测结果，包含检测框和置信度标签

终端输出:
    物品3D坐标，如 white_part(0.123, -0.045, 0.456)
"""

import sys
import os
import cv2
import numpy as np
import rospy
import struct
import zlib
from cv_bridge import CvBridge
import tf.transformations as tr
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

try:
    from ultralytics import YOLO
except ImportError:
    rospy.logerr("❌ 未安装 ultralytics, 请执行: pip install ultralytics")
    sys.exit(1)


import threading

# ============================ 配置参数 ============================
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
MODEL_PATH = "src/yolov8_detection/scripts/models/best2.pt"
IMAGE_TOPIC = "/cam_h/color/image_raw"
DEPTH_TOPIC = "/cam_h/depth/image_raw/compressedDepth"
CAMERA_INFO_TOPIC = "/cam_h/color/camera_info"
SHOW_WINDOW = True # 是否显示检测窗口


# ============================ 零件检测器 ============================

class YoloPartsDetector:
    def __init__(self):
        rospy.init_node('yolo_parts_detector', anonymous=True)

        model_path = rospy.get_param('~model', MODEL_PATH)
        self.show_window = rospy.get_param('~show_window', SHOW_WINDOW)
        if self.show_window and os.environ.get('DISPLAY') is None:
            rospy.logwarn("⚠️ 未检测到 DISPLAY 环境变量，将禁用窗口显示")
            self.show_window = False
        
        try:
            self.model = YOLO(model_path)
            rospy.loginfo(f"✅ YOLO 模型加载成功: {model_path}")
        except Exception as e:
            rospy.logerr(f"❌ 加载模型失败: {e}")
            sys.exit(1)

        self.bridge = CvBridge()

        self.depth_image = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False
        self.depth_received = False
        
        self.current_annotated_frame = None
        self.frame_lock = threading.Lock()

        self.translation = [0.092, -0.000, 0.670]
        self.quaternion = [-0.614, 0.614, -0.350, 0.350]

        self.T_cam_to_base = tr.quaternion_matrix(self.quaternion)
        self.T_cam_to_base[0:3, 3] = self.translation

        rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        rospy.Subscriber(DEPTH_TOPIC, CompressedImage, self.depth_callback, queue_size=1)
        rospy.Subscriber(CAMERA_INFO_TOPIC, CameraInfo, self.camera_info_callback, queue_size=1)

        self.detection_pub = rospy.Publisher('/yolo_parts/detections', PointStamped, queue_size=10)

        rospy.loginfo("🎯 YOLOv8 黑白零件检测器已启动")
        rospy.loginfo(f"📷 订阅图像: {IMAGE_TOPIC}")
        rospy.loginfo(f"📷 订阅深度: {DEPTH_TOPIC}")
        rospy.loginfo(f"📷 订阅相机内参: {CAMERA_INFO_TOPIC}")
        rospy.loginfo(f"📊 置信度阈值: {CONF_THRESHOLD}")
        if self.show_window:
            rospy.loginfo("🖼️  显示检测窗口 (按 'q' 键退出)")

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            self.camera_info_received = True
            rospy.loginfo(f"📐 相机内参: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def depth_callback(self, msg):
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            depth_img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if depth_img is None and len(buf) > 12:
                depth_img = cv2.imdecode(buf[12:], cv2.IMREAD_UNCHANGED)
            if depth_img is None and len(buf) > 16:
                depth_img = cv2.imdecode(buf[16:], cv2.IMREAD_UNCHANGED)

            if depth_img is not None:
                self.depth_image = depth_img
                if not self.depth_received:
                    self.depth_received = True
                    rospy.loginfo(f"✅ 深度图像接收/解码成功, 尺寸: {depth_img.shape}")
        except Exception as e:
            rospy.logerr(f"深度图像处理异常: {e}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 增加推理前后的日志，用于排查崩溃点
            # rospy.logdebug("开始 YOLO 推理...")
            try:
                results = self.model(frame, verbose=False,
                                    conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            except Exception as e:
                rospy.logerr(f"YOLO 推理过程中崩溃: {e}")
                return
            # rospy.logdebug("YOLO 推理完成")

            annotated_frame, parts_info = self._draw_detections(frame, results)

            if parts_info:
                for name, x, y, z, bx, by, bz in parts_info:
                    rospy.loginfo(f"📦 {name} | Cam: ({x:.3f}, {y:.3f}, {z:.3f}) | BaseLink: ({bx:.3f}, {by:.3f}, {bz:.3f})")

                    det_msg = PointStamped()
                    det_msg.header = Header()
                    det_msg.header.stamp = rospy.Time.now()
                    det_msg.header.frame_id = name
                    det_msg.point.x = bx
                    det_msg.point.y = by
                    det_msg.point.z = bz
                    self.detection_pub.publish(det_msg)
            else:
                rospy.loginfo_throttle(5.0, "👀 监控中，未检测到零件或无法获取深度...")

            if self.show_window:
                with self.frame_lock:
                    self.current_annotated_frame = annotated_frame

        except Exception as e:
            rospy.logerr(f"图像处理异常: {e}")
            import traceback; traceback.print_exc()

    def _pixel_to_3d(self, u, v):
        if self.depth_image is None:
            return None

        if self.fx is None:
            return None

        try:
            if len(self.depth_image.shape) == 3:
                depth = int(self.depth_image[v, u, 0])
            else:
                depth = int(self.depth_image[v, u])

            depth_m = depth / 1000.0
        except Exception as e:
            return None

        if depth_m <= 0 or depth_m > 10:
            return None

        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m

        return x, y, z

    def _transform_to_base_link(self, x_cam, y_cam, z_cam):
        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_base = np.dot(self.T_cam_to_base, pt_cam)
        return pt_base[0], pt_base[1], pt_base[2]

    def _draw_detections(self, frame, results):
        annotated_frame = frame.copy()
        parts_info = []

        detection_count = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = result.names[cls]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                coords_3d = self._pixel_to_3d(cx, cy)

                if coords_3d:
                    px, py, pz = coords_3d
                    bx, by, bz = self._transform_to_base_link(px, py, pz)
                    parts_info.append((name, px, py, pz, bx, by, bz))
                    display_text = f"{name} {conf:.2f} ({bx:.3f}, {by:.3f}, {bz:.3f})"
                else:
                    display_text = f"{name} {conf:.2f}"

                detection_count += 1

                color = (255, 0, 0)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_width, label_height = label_size

                cv2.rectangle(annotated_frame,
                             (x1, y1 - label_height - 8),
                             (x1 + label_width + 8, y1),
                             color,
                             cv2.FILLED)

                cv2.putText(annotated_frame, display_text,
                            (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1)

        if detection_count > 0 and parts_info:
            rospy.loginfo(f"🎯 检测到 {detection_count} 个目标, 其中 {len(parts_info)} 个有3D坐标")
        elif detection_count > 0 and not parts_info:
            rospy.loginfo(f"🎯 检测到 {detection_count} 个目标, 但无法获取3D坐标")

        return annotated_frame, parts_info

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.show_window:
                display_frame = None
                with self.frame_lock:
                    if self.current_annotated_frame is not None:
                        display_frame = self.current_annotated_frame.copy()
                        self.current_annotated_frame = None
                
                if display_frame is not None:
                    try:
                        cv2.imshow("YOLO Parts Detection", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        rospy.logerr(f"可视化窗口异常: {e}")
                        self.show_window = False
            rate.sleep()

    def __del__(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        detector = YoloPartsDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLO 零件检测器已停止")
    except Exception as e:
        rospy.logerr(f"启动失败: {e}")
        import traceback; traceback.print_exc()
