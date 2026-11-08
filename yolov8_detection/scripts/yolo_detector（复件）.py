#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测节点 - 使用 best.pt 进行检测并显示置信度
支持仿真环境的压缩图像话题
"""

import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from kuavo_msgs.msg import yoloDetection, yoloOutputData
import os


class YOLODetector:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)
        
        # 1. 找到 best.pt 模型
        self.model_path = self.find_model_path()
        
        # 2. 加载模型
        rospy.loginfo(f"正在加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        rospy.loginfo("✅ 模型加载成功！")
        
        # 3. 初始化相机和图像处理
        self.bridge = CvBridge()
        self.use_compressed = False  # 是否使用压缩图像
        
        # 4. 设置参数
        self.confidence_threshold = rospy.get_param('~conf', 0.6)  # 默认置信度阈值
        self.show_image = rospy.get_param('~show', True)           # 是否显示图像
        
        # 5. 自动检测并订阅相机图像
        self.auto_detect_camera_topic()
        
        # 6. 发布检测结果
        self.detection_pub = rospy.Publisher(
            '/yolo_detections', 
            yoloDetection, 
            queue_size=10
        )
        
        rospy.loginfo(f"📷 正在订阅相机话题: {self.camera_topic}")
        rospy.loginfo(f"🎯 置信度阈值: {self.confidence_threshold}")
        
    def find_model_path(self):
        """
        查找 best.pt 模型文件
        """
        # 搜索路径列表（按优先级）
        search_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best.pt'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt'),
            '/home/y/桌面/leju_kuavo_craic_2026/src/demo/yolo_object_capture/models/best.pt',
            '/home/y/桌面/leju_kuavo_craic_2026/src/craic_task_template/scripts/models/best.pt',
            '/home/y/桌面/leju_kuavo_craic_2026/best.pt',
            '/home/y/best.pt',
            '/root/kuavo_ws/src/craic_task_template/scripts/models/best.pt',
            '/root/kuavo_ws/src/craic_task_template/scripts/best.pt',
            '/root/kuavo_ws/best.pt',
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                rospy.loginfo(f"找到模型文件: {path}")
                return path
        
        # 如果没找到，使用默认的 yolov8n 模型
        rospy.logwarn("⚠️  未找到 best.pt，将使用默认的 yolov8n.pt")
        return 'yolov8n.pt'
    
    def auto_detect_camera_topic(self):
        """
        自动检测可用的相机话题
        优先使用仿真环境的压缩图像话题
        """
        # 仿真环境相机话题优先级列表
        priority_topics = [
            '/cam_h/color/image_raw/compressed',   # 头部相机（压缩）
            '/cam_l/color/image_raw/compressed',   # 左手腕相机（压缩）
            '/cam_r/color/image_raw/compressed',   # 右手腕相机（压缩）
            '/camera/color/image_raw',             # 默认普通图像话题
            '/camera/image_raw',
            '/kuavo/camera/image_raw',
        ]
        
        # 获取当前发布的所有话题
        try:
            import rosgraph
            master = rosgraph.Master('/yolo_detector')
            topics = master.getPublishedTopics('/')
            topic_list = [t[0] for t in topics]
            
            # 按优先级查找可用话题
            found = False
            for topic in priority_topics:
                if topic in topic_list:
                    self.camera_topic = topic
                    self.use_compressed = topic.endswith('/compressed')
                    rospy.loginfo(f"🎥 自动检测到相机话题: {self.camera_topic}")
                    found = True
                    break

            if not found:
                # 如果没找到，使用默认话题
                rospy.logwarn(f"⚠️  未检测到相机话题，使用默认: {priority_topics[0]}")
                self.camera_topic = priority_topics[0]
                self.use_compressed = True
            
        except Exception as e:
            rospy.logwarn(f"⚠️  无法检测话题，使用默认设置: {e}")
            self.camera_topic = '/cam_h/color/image_raw/compressed'
            self.use_compressed = True
        
        # 订阅相机话题
        if self.use_compressed:
            self.image_sub = rospy.Subscriber(
                self.camera_topic,
                CompressedImage,
                self.compressed_image_callback,
                queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                self.camera_topic,
                Image,
                self.image_callback,
                queue_size=1
            )
    
    def compressed_image_callback(self, msg):
        """
        处理压缩图像回调
        """
        try:
            # 将压缩图像转换为 OpenCV 格式
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"压缩图像解码错误: {e}")
            return
        
        # 进行 YOLO 推理
        results = self.model.predict(
            cv_image, 
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # 处理结果
        self.process_results(results, cv_image)
    
    def image_callback(self, img_msg):
        """
        处理普通图像回调
        """
        try:
            # 转换 ROS 图像为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误: {e}")
            return
        
        # 进行 YOLO 推理
        results = self.model.predict(
            cv_image, 
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # 处理结果
        self.process_results(results, cv_image)
        
    def process_results(self, results, cv_image):
        """
        处理检测结果
        """
        msg = yoloDetection()
        msg.header.stamp = rospy.Time.now()

        detection_list = []

        # 遍历检测结果
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    det = yoloOutputData()

                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    det.x_pos = float(x1)
                    det.y_pos = float(y1)
                    det.width = float(x2 - x1)
                    det.height = float(y2 - y1)

                    # 类别 ID 和名称
                    det.class_id = int(box.cls[0].item())
                    det.class_name = self.model.names.get(det.class_id, str(det.class_id))

                    # 置信度
                    det.confidence = float(box.conf[0].item())

                    detection_list.append(det)

        msg.data = detection_list

        # 发布结果
        self.detection_pub.publish(msg)
        
        # 打印检测信息（显示置信度）
        if detection_list:
            self.print_detections(detection_list)
        
        # 显示图像和检测框
        if self.show_image:
            annotated_frame = results[0].plot()  # 使用 ultralytics 自带的绘制
            cv2.imshow('YOLO Detection (Press Q to quit)', annotated_frame)
            cv2.waitKey(1)
            
    def print_detections(self, detections):
        """
        美化打印检测结果（包含置信度）
        """
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"检测到 {len(detections)} 个目标:")
        
        for i, det in enumerate(detections):
            conf_percent = det.confidence * 100
            rospy.loginfo(f"目标 {i+1}: 类别={det.class_name}({det.class_id}), "
                         f"置信度={conf_percent:.1f}%, "
                         f"位置=({det.x_pos:.0f}, {det.y_pos:.0f}), "
                         f"大小={det.width:.0f}x{det.height:.0f}")
        
        rospy.loginfo("=" * 60)

    def run(self):
        rospy.loginfo("🎉 YOLO 检测器启动成功！")
        rospy.spin()
        
        # 关闭窗口
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        detector = YOLODetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("检测节点已停止")
