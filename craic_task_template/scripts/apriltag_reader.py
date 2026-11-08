#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 识别节点 - 检测料盘架上的二维码并获取相对坐标
订阅 apriltag_ros 发布的 /tag_detections 话题
支持将坐标转换到 base_link 坐标系
"""

import rospy
import math
import numpy as np
import tf
import tf2_ros
import tf.transformations as tr
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import Pose, PoseStamped


class AprilTagReader:
    def __init__(self):
        rospy.init_node('apriltag_reader', anonymous=True)

        self.tag_positions = {}
        self.target_ids = [7, 8]  # 料盘架上的二维码ID

        # TF 监听器
        self.tf_listener = tf.TransformListener()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener2 = tf2_ros.TransformListener(self.tf_buffer)

        # 尝试多个可能的相机坐标系名称
        self.camera_frames = [
            'Head Camera View',             # MuJoCo 仿真 Kuavo 52 头部相机
            'camera_color_optical_frame',   # 标准 Realsense 光学系
            'camera_color_frame',           # 标准 Realsense
            'cam_h_color_optical_frame',    # 仿真头部相机
            'camera_link',                  # 通用相机链接
            'head_camera_frame',            # 头部相机
            'camera_rgb_frame',             # 旧版 Realsense
            'cam_h_rgb_frame',              # 仿真 cam_h
        ]

        self.camera_frame = None  # 将在首次检测时确定

        # 订阅 apriltag_ros 发布的原始检测结果
        self.tag_sub = rospy.Subscriber(
            '/tag_detections',
            AprilTagDetectionArray,
            self.tag_callback,
            queue_size=1
        )

        rospy.loginfo("=" * 60)
        rospy.loginfo("🎯 AprilTag 识别节点启动成功！")
        rospy.loginfo("   订阅话题: /tag_detections")
        rospy.loginfo("   监控二维码 ID: 7 (第三层), 8 (第二层)")
        rospy.loginfo("=" * 60)

    def detect_camera_frame(self, msg):
        """
        从消息头中检测相机坐标系名称
        """
        if self.camera_frame is not None:
            return

        if msg.header.frame_id:
            rospy.loginfo(f"检测到相机帧名: {msg.header.frame_id}")
            self.camera_frame = msg.header.frame_id

    def transform_point_to_base_link(self, x, y, z, source_frame):
        """
        将相机坐标系下的3D点转换到 base_link 坐标系
        """
        # 优先使用检测消息自带的 frame_id，再遍历已知列表
        frames_to_try = []
        if source_frame:
            frames_to_try.append(source_frame)
        if self.camera_frame and self.camera_frame not in frames_to_try:
            frames_to_try.append(self.camera_frame)
        for f in self.camera_frames:
            if f not in frames_to_try:
                frames_to_try.append(f)

        for frame in frames_to_try:
            try:
                (trans, rot) = self.tf_listener.lookupTransform(
                    'base_link',
                    frame,
                    rospy.Time(0)
                )

                # 构建变换矩阵
                T = tr.quaternion_matrix(rot)
                T[0:3, 3] = trans

                # 相机系下的点 (齐次坐标)
                cam_point = np.array([x, y, z, 1.0])

                # 执行变换
                base_point = np.dot(T, cam_point)

                rospy.loginfo(f"使用 TF 帧: {frame}")
                return (base_point[0], base_point[1], base_point[2])

            except Exception as e:
                continue

        # 如果所有 TF 都失败，使用近似变换
        rospy.logwarn("所有 TF 变换失败，使用近似变换")

        # 近似变换 (基于 URDF 的头部结构)
        # 头部相机大约在 base_link 上方 0.65m，前方 0.05m
        base_x = x + 0.05   # 相机在前方
        base_y = y - 0.02   # Y 轴偏移
        base_z = z + 0.65   # 加上头部高度

        return (base_x, base_y, base_z)

    def tag_callback(self, msg):
        """
        处理 AprilTag 检测结果
        """
        # 检测相机帧名
        self.detect_camera_frame(msg)

        if not msg.detections:
            return

        rospy.loginfo("=" * 60)
        rospy.loginfo(f"检测到 {len(msg.detections)} 个 AprilTag")

        for detection in msg.detections:
            # AprilTag ID (可能是数组，取第一个)
            tag_id = detection.id[0] if len(detection.id) > 0 else -1

            if tag_id not in self.target_ids:
                continue

            # 获取位姿 (PoseWithCovarianceStamped)
            pose_with_cov = detection.pose
            pose = pose_with_cov.pose.pose

            x = pose.position.x
            y = pose.position.y
            z = pose.position.z

            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w

            roll, pitch, yaw = self.quaternion_to_euler(qx, qy, qz, qw)

            # 转换到 base_link 坐标系
            # 优先使用单个检测的 pose frame_id，其次使用消息级 header
            det_frame = pose_with_cov.header.frame_id or msg.header.frame_id
            base_pos = self.transform_point_to_base_link(x, y, z, det_frame)

            layer_name = "第三层" if tag_id == 7 else "第二层"

            rospy.loginfo(f"")
            rospy.loginfo(f"📦 AprilTag ID={tag_id} ({layer_name})")
            rospy.loginfo(f"   ├─ 相机坐标系位置:")
            rospy.loginfo(f"   │    x = {x:.3f} m")
            rospy.loginfo(f"   │    y = {y:.3f} m")
            rospy.loginfo(f"   │    z = {z:.3f} m (距离相机的深度)")
            rospy.loginfo(f"   ├─ base_link 坐标系位置:")
            rospy.loginfo(f"   │    x = {base_pos[0]:.3f} m (前后方向，正=前)")
            rospy.loginfo(f"   │    y = {base_pos[1]:.3f} m (左右方向，正=左)")
            rospy.loginfo(f"   │    z = {base_pos[2]:.3f} m (上下方向，正=上)")
            rospy.loginfo(f"   ├─ 相机系旋转角度:")
            rospy.loginfo(f"   │    Roll  = {math.degrees(roll):.1f}°")
            rospy.loginfo(f"   │    Pitch = {math.degrees(pitch):.1f}°")
            rospy.loginfo(f"   │    Yaw   = {math.degrees(yaw):.1f}°")

            # 计算距离
            distance_cam = math.sqrt(x**2 + y**2 + z**2)
            distance_base = math.sqrt(base_pos[0]**2 + base_pos[1]**2 + base_pos[2]**2)

            rospy.loginfo(f"   ├─ 距离相机: {distance_cam:.3f} m")
            rospy.loginfo(f"   └─ 距离 base_link 原点: {distance_base:.3f} m")

            self.tag_positions[tag_id] = {
                'camera': {'x': x, 'y': y, 'z': z},
                'base_link': {'x': base_pos[0], 'y': base_pos[1], 'z': base_pos[2]},
                'distance': distance_base
            }

        rospy.loginfo("=" * 60)

    def quaternion_to_euler(self, qx, qy, qz, qw):
        """
        将四元数转换为欧拉角 (roll, pitch, yaw)
        """
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def get_tag_position(self, tag_id):
        """
        获取指定ID的二维码位置
        """
        return self.tag_positions.get(tag_id)

    def get_distance_to_tag(self, tag_id):
        """
        计算 base_link 到指定二维码的距离
        """
        pos = self.get_tag_position(tag_id)
        if pos:
            return pos.get('distance')
        return None

    def run(self):
        rospy.loginfo("等待 AprilTag 检测...")
        rospy.spin()


def main():
    reader = AprilTagReader()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("AprilTag 识别节点已停止")


if __name__ == '__main__':
    main()
