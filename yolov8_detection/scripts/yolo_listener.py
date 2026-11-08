#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测结果监听器 - 专门用于显示置信度
"""

import rospy
from kuavo_msgs.msg import yoloDetection


class YOLOListener:
    def __init__(self):
        rospy.init_node('yolo_listener', anonymous=True)
        
        self.sub = rospy.Subscriber(
            '/yolo_detections', 
            yoloDetection, 
            self.callback, 
            queue_size=10
        )
        
        rospy.loginfo("🎯 YOLO 检测结果监听器已启动")
        rospy.loginfo("📊 正在等待检测结果（包含置信度）...")
        
    def callback(self, msg):
        rospy.loginfo("-" * 70)
        
        if len(msg.data) == 0:
            rospy.loginfo("❌ 未检测到任何目标")
            return
            
        rospy.loginfo(f"✅ 检测到 {len(msg.data)} 个目标:")
        
        for i, det in enumerate(msg.data):
            confidence = det.confidence
            conf_percent = confidence * 100
            
            # 根据置信度设置颜色/标记
            if conf_percent >= 90:
                status = "🌟 (高置信度)"
            elif conf_percent >= 70:
                status = "✅ (中置信度)"
            elif conf_percent >= 50:
                status = "⚠️  (低置信度)"
            else:
                status = "❌ (极低置信度)"
            
            rospy.loginfo(f"   目标 {i+1}:")
            rospy.loginfo(f"      类别: {det.class_name} (ID={det.class_id})")
            rospy.loginfo(f"      置信度: {conf_percent:.2f}% {status}")
            rospy.loginfo(f"      位置: ({det.x_pos:.0f}, {det.y_pos:.0f})")
            rospy.loginfo(f"      大小: {det.width:.0f} x {det.height:.0f}")
            
            # 输出调试信息
            rospy.loginfo(f"      原始置信度值: {det.confidence:.6f}")
            
        rospy.loginfo("-" * 70)
        
    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        listener = YOLOListener()
        listener.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("监听器已停止")
