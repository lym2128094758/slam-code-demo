#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取控制器 v4 — YOLO 视觉+IK 精准抓取

完整流程 (auto=True):
  Phase 1: 检测 AprilTag 获取货架位置
  Phase 2: 行走到合适抓取距离
  Phase 3: 使用 YOLO 检测坐标 + IK 求解执行抓取

改进要点：
1. 优先使用 YOLO 检测的完整3D坐标（x, y, z）
2. 二维码仅用于定位货架和导航
3. 优化 IK 参数，提高求解精度和稳定性
4. 添加坐标验证和容错机制

订阅:
    /tag_detections        — AprilTag 检测（用于导航定位）
    /yolo_best_target      — YOLO 最佳目标（base_link 坐标系）
    /sensors_data_raw      — 传感器数据（手臂关节反馈）

控制接口:
    /arm_traj_change_mode         — 手臂控制模式
    /ik/two_arm_hand_pose_cmd_srv — IK 逆解服务
    /kuavo_arm_target_poses       — 关节角目标
    /control_robot_hand_position  — 夹爪控制
    /robot_head_motion_data       — 头部控制
    /cmd_vel                      — 行走速度
"""

import sys
import os
import math
import time
import numpy as np
import rospy
import tf
import tf.transformations as tr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../craic_simulator/utils'))
from gripper_controller import GripperController

from geometry_msgs.msg import Twist, PointStamped
from kuavo_msgs.msg import (
    armTargetPoses,
    robotHandPosition,
    robotHeadMotionData,
    twoArmHandPoseCmd,
    ikSolveParam,
    sensorsData,
)
from kuavo_msgs.srv import (
    changeArmCtrlMode,
    changeArmCtrlModeRequest,
    twoArmHandPoseCmdSrv,
)

try:
    from apriltag_ros.msg import AprilTagDetectionArray
except ImportError:
    from kuavo_msgs.msg import AprilTagDetectionArray


# ============================ 常量 / 配置 ============================

# 夹爪控制值 (1=完全闭合, 0=完全张开)
CLOSE_HAND = 0.8  # 夹爪闭合程度
OPEN_HAND  = 0.0

# 双臂 home 关节角（度），左 7 + 右 7
ARM_HOME_DEG = [
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
]

# 52 机器人零位 (base_link)
ROBOT_ZERO_52 = (-0.012, -0.225, -0.315)

# 抓取安全范围 (base_link)
MIN_REACH_X =  0.15
MAX_REACH_X =  0.70
MIN_HEIGHT_Z = -0.50
MAX_HEIGHT_Z =  0.70

# AprilTag 已知世界坐标 (craic_room.yaml)
TAG_WORLD_POS = {
    8: np.array([-0.25, -1.80, 0.945]),   # 第二层架板
    7: np.array([-0.25, -1.80, 1.445]),   # 第三层架板
}

# 导航参数
NAV_TARGET_DISTANCE = 0.35   # 到货架前表面的期望距离 (m)
NAV_TOLERANCE       = 0.015   # 到达容差 (m)
NAV_MAX_VEL         = 0.1   # 最大行走线速度 (m/s)
NAV_TIMEOUT         = 20.0   # 导航超时 (s)

# 抓取深度偏移：物体与二维码贴同一竖面，前向额外伸入距离
GRASP_DEPTH_OFFSET = 0.18  # 增加到18cm，让手臂伸得更深到货架里面

# 层板物体相对 AprilTag 的垂直偏移（当无YOLO时使用）
TAG_LAYER_OFFSET = {
    7: -0.3,   # 上层：二维码下方约0.3m
    8: -0.2,   # 下层：二维码下方约0.2m
}

# 抓取高度下限保护
MIN_GRASP_HEIGHT = 0.22

# MuJoCo 仿真中头部相机可能的 TF frame 名
CAMERA_FRAMES = [
    'Head Camera View',             # MuJoCo Kuavo 52
    'camera_color_optical_frame',   # Realsense
    'cam_h_color_optical_frame',    # 仿真 cam_h
]

# YOLO 数据有效期（秒）
YOLO_VALID_DURATION = 2.0


# ============================ 工具函数 ============================

class Q:
    """轻量四元数容器"""
    __slots__ = ('w', 'x', 'y', 'z')
    def __init__(self): self.w = self.x = self.y = self.z = 0.0


def euler_to_quat(yaw, pitch, roll=0.0):
    """Rz(yaw) * Ry(pitch) * Rx(roll) → 四元数"""
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp_, sp_ = math.cos(pitch), math.sin(pitch)
    cr_, sr_ = math.cos(roll), math.sin(roll)
    
    q = Q()
    q.w = cy_ * cp_ * cr_ + sy_ * sp_ * sr_
    q.x = cy_ * cp_ * sr_ - sy_ * sp_ * cr_
    q.y = sy_ * cp_ * sr_ + cy_ * sp_ * cr_
    q.z = sy_ * cp_ * cr_ - cy_ * sp_ * sr_
    
    n = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    if n > 0:
        q.w /= n; q.x /= n; q.y /= n; q.z /= n
    return q


def make_ik_solve_param(strict=False):
    """【优化】IK 求解参数 - 根据需求调整严格程度"""
    p = ikSolveParam()
    if strict:
        # 严格模式：高精度要求
        p.major_optimality_tol   = 1e-3
        p.major_feasibility_tol  = 1e-3
        p.minor_feasibility_tol  = 1e-3
        p.major_iterations_limit = 1000
        p.oritation_constraint_tol = 1e-3
        p.pos_constraint_tol     = 1e-3
        p.pos_cost_weight        = 0.5
    else:
        # 宽松模式：提高求解成功率
        p.major_optimality_tol   = 1e-2   # 放宽到1厘米
        p.major_feasibility_tol  = 1e-2
        p.minor_feasibility_tol  = 1e-2
        p.major_iterations_limit = 500
        p.oritation_constraint_tol = 0.3   # 大幅放宽姿态约束，优先保证位置精度
        p.pos_constraint_tol     = 5e-3
        p.pos_cost_weight        = 0.7  # 更重视位置
    return p


# ============================ 主控制器 ============================

class GrabController:
    def __init__(self):
        rospy.init_node('grab_controller', anonymous=True)

        self.is_grabbing = False

        # ---- ROS 参数 ----
        self.auto_grab      = rospy.get_param('~auto', False)
        self.target_tag_id  = int(rospy.get_param('~target_tag', 7))
        self.head_pitch_deg = float(rospy.get_param('~head_pitch', 20.0))
        self.nav_target_dist = float(rospy.get_param('~nav_dist', NAV_TARGET_DISTANCE))

        # ---- TF 监听器 ----
        self.tf_listener = tf.TransformListener()

        # ---- 发布者 ----
        self.head_pub    = rospy.Publisher('/robot_head_motion_data', robotHeadMotionData, queue_size=10)
        self.arm_pub     = rospy.Publisher('/kuavo_arm_target_poses', armTargetPoses, queue_size=10)
        self.hand_pub    = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # ---- 夹爪控制器 ----
        self.gripper = GripperController()

        # ---- AprilTag 缓存 ----
        self.latest_tags = {}
        self.last_tag_msg_time = None
        self.last_seen_tag_ids = set()
        self.last_tf_fail_tag_ids = set()
        rospy.Subscriber('/tag_detections', AprilTagDetectionArray,
                         self._tag_raw_cb, queue_size=1)

        # ---- YOLO 最佳目标缓存 ----
        self.latest_yolo = None
        rospy.Subscriber('/yolo_best_target', PointStamped, self._yolo_cb, queue_size=1)

        # ---- 传感器数据订阅 ----
        self.current_sensors_data = None
        rospy.Subscriber('/sensors_data_raw', sensorsData, self._sensors_cb, queue_size=1)

        # 等待手臂控制可用
        rospy.loginfo("⏳ 等待 /kuavo_arm_target_poses 订阅者...")
        rate = rospy.Rate(10)
        t0 = rospy.Time.now()
        while self.arm_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            if (rospy.Time.now() - t0).to_sec() > 10.0:
                rospy.logwarn("⚠️  10s 无订阅者, 继续启动")
                break
            rate.sleep()

        layer = "上层" if self.target_tag_id == 7 else "下层"
        rospy.loginfo(
            f"🎯 grab_controller v4 启动 | auto={self.auto_grab} "
            f"target_tag={self.target_tag_id}({layer}) "
            f"nav_dist={self.nav_target_dist:.2f}m head_pitch={self.head_pitch_deg}°"
        )

    # ==================== TF 变换 ====================

    def _transform_to_base_link(self, x, y, z, source_frame):
        """将相机坐标系下的点变换到 base_link"""
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
                return pt[0], pt[1], pt[2]
            except Exception:
                continue
        return None

    def _tag_raw_cb(self, msg):
        """订阅 /tag_detections，TF 变换到 base_link 后缓存"""
        self.last_tag_msg_time = rospy.Time.now()
        for det in msg.detections:
            tag_id = det.id[0]
            self.last_seen_tag_ids.add(tag_id)
            pose = det.pose.pose.pose
            src_frame = det.pose.header.frame_id or msg.header.frame_id or ''

            result = self._transform_to_base_link(
                pose.position.x, pose.position.y, pose.position.z, src_frame)
            if result is None:
                self.last_tf_fail_tag_ids.add(tag_id)
                rospy.logwarn_throttle(5.0, f"TF 变换失败 tag {tag_id}, 跳过")
                continue

            bx, by, bz = result
            self.latest_tags[tag_id] = {
                'x': bx, 'y': by, 'z': bz,
                'stamp': rospy.Time.now(),
            }
            rospy.loginfo_throttle(
                3.0,
                f"📌 tag {tag_id} base_link: x={bx:.3f} y={by:.3f} z={bz:.3f}"
            )

    def _yolo_cb(self, msg):
        """订阅 /yolo_best_target（PointStamped in base_link）"""
        try:
            x = float(msg.point.x)
            y = float(msg.point.y)
            z = float(msg.point.z)
            stamp = msg.header.stamp if msg.header and msg.header.stamp else rospy.Time.now()
            self.latest_yolo = {'x': x, 'y': y, 'z': z, 'stamp': stamp}
            rospy.loginfo_throttle(2.0, f"🎯 YOLO best base_link: x={x:.3f} y={y:.3f} z={z:.3f}")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"YOLO 回调解析失败: {e}")

    def _sensors_cb(self, msg):
        """订阅传感器数据"""
        self.current_sensors_data = msg

    # ==================== 等待手臂到位 ====================

    def wait_for_arm_arrival(self, target_joints_deg, timeout=15.0, tolerance=15.0, min_wait=0.0):
        """等待手臂到达目标位置"""
        if min_wait > 0:
            rospy.loginfo(f"⏳ 等待手臂运动完成 ({min_wait:.1f}s)...")
            time.sleep(min_wait)

        rospy.loginfo(f"⏳ 检查手臂是否到位 (超时={timeout}s, 容差={tolerance}°)")
        start_time = rospy.Time.now()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()

            if elapsed > timeout:
                rospy.logwarn(f"⚠️  手臂到位超时 ({timeout}s)")
                return False

            if self.current_sensors_data is None:
                rospy.loginfo_throttle(2.0, "⏳ 等待 /sensors_data_raw...")
                rate.sleep()
                continue

            try:
                arm_q = self.current_sensors_data.joint_data.joint_q[13:27]
                max_error = max(
                    abs(math.degrees(arm_q[i]) - target_joints_deg[i])
                    for i in range(14)
                )
                rospy.loginfo_throttle(2.0,
                    f"⏳ 手臂误差={max_error:.2f}° (已等 {elapsed:.1f}s)")

                if max_error <= tolerance:
                    rospy.loginfo(f"✅ 手臂到位! 误差={max_error:.2f}°, 用时={elapsed:.1f}s")
                    return True

            except Exception as e:
                rospy.logwarn_throttle(2.0, f"关节读取异常: {e}")

            rate.sleep()

        return False

    # ==================== 基础控制原语 ====================

    def set_head(self, yaw_deg=0.0, pitch_deg=0.0):
        msg = robotHeadMotionData()
        msg.joint_data = [float(yaw_deg), float(pitch_deg)]
        self.head_pub.publish(msg)
        rospy.loginfo(f"🙂 head → yaw={yaw_deg:.1f} pitch={pitch_deg:.1f}")

    def set_arm_control_mode(self, mode):
        try:
            rospy.wait_for_service('/arm_traj_change_mode', timeout=5.0)
            cli = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            req = changeArmCtrlModeRequest(); req.control_mode = mode
            resp = cli(req)
            if resp.result:
                rospy.loginfo(f"✅ arm mode → {mode}")
                return True
            rospy.logwarn(f"⚠️  arm mode 失败: {resp.message}")
            return False
        except Exception as e:
            rospy.logerr(f"❌ arm mode 服务异常: {e}")
            return False

    def call_ik(self, eef_pose_msg):
        try:
            rospy.wait_for_service('/ik/two_arm_hand_pose_cmd_srv', timeout=5.0)
            srv = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
            return srv(eef_pose_msg)
        except Exception as e:
            rospy.logerr(f"❌ IK 异常: {e}")
            return None

    def publish_arm_target(self, times, values_deg):
        msg = armTargetPoses()
        msg.times  = list(times)
        msg.values = list(values_deg)
        self.arm_pub.publish(msg)

    def stop_walking(self):
        vel = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(vel)
            rospy.sleep(0.05)

    def forward_goal(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=1.0):
        """发布速度指令让机器人移动"""
        vel = Twist()
        vel.linear.x = linear_x
        vel.linear.y = linear_y
        vel.angular.z = angular_z
        
        rate = rospy.Rate(20)
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            self.cmd_vel_pub.publish(vel)
            rate.sleep()
        
        self.stop_walking()

    # ==================== Phase 1: AprilTag 检测 + 导航 ====================

    def get_fresh_tag(self, tag_id, max_age=2.0):
        """获取指定 tag 的最新 base_link 数据"""
        if tag_id in self.latest_tags:
            tag = self.latest_tags[tag_id]
            if (rospy.Time.now() - tag['stamp']).to_sec() < max_age:
                return tag
        return None

    def wait_for_tag(self, tag_id, timeout=10.0):
        """等待指定 tag 被检测到"""
        rospy.loginfo(f"🔍 等待 AprilTag {tag_id} 检测...")
        t0 = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < timeout:
            tag = self.get_fresh_tag(tag_id)
            if tag is not None:
                rospy.loginfo(
                    f"✅ 检测到 tag {tag_id}: "
                    f"x={tag['x']:.3f} y={tag['y']:.3f} z={tag['z']:.3f}")
                return tag
            rate.sleep()
        rospy.logwarn(f"⚠️  等待 tag {tag_id} 超时 ({timeout}s)")
        if self.last_tag_msg_time is None:
            rospy.logwarn("⚠️  超时期间完全没有收到 /tag_detections，说明 AprilTag 检测节点未启动或未发布")
        else:
            age = (rospy.Time.now() - self.last_tag_msg_time).to_sec()
            rospy.logwarn(f"⚠️  最近一次 /tag_detections 距现在 {age:.1f}s")
            if self.last_seen_tag_ids:
                rospy.logwarn(f"⚠️  最近收到过的 tag id: {sorted(self.last_seen_tag_ids)}")
            if tag_id in self.last_tf_fail_tag_ids:
                rospy.logwarn(f"⚠️  tag {tag_id} 曾被检测到，但 TF 变换到 base_link 失败")
        return None

    def navigate_to_distance(self, tag_id, target_dist, timeout=NAV_TIMEOUT):
        """导航到目标距离"""
        rospy.loginfo(f"🚶 导航 → tag {tag_id}, 目标 x≈{target_dist:.2f}m")

        t0 = rospy.Time.now()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < timeout:
            tag = self.get_fresh_tag(tag_id)
            if tag is None:
                rospy.loginfo_throttle(2.0, "🔍 tag 丢失, 等待...")
                rate.sleep()
                continue

            dist  = tag['x']
            error = dist - target_dist
            rospy.loginfo_throttle(
                1.0,
                f"📐 tag {tag_id}: x={dist:.3f}m → target={target_dist:.2f}m  err={error:+.3f}m")

            if abs(error) < NAV_TOLERANCE:
                self.stop_walking()
                rospy.loginfo(f"✅ 到位: x={dist:.3f}m ≈ {target_dist:.2f}m")
                return True

            vel = Twist()
            vel.linear.x = max(-NAV_MAX_VEL, min(NAV_MAX_VEL, error * 0.5))
            self.cmd_vel_pub.publish(vel)
            rate.sleep()

        self.stop_walking()
        rospy.logwarn(f"⚠️  导航超时 ({timeout}s)")
        return False

    # ==================== 坐标融合 ====================

    def get_grasp_target(self, tag_id):
        """
        获取抓取目标坐标（优先使用 YOLO）
        返回: (gx, gy, gz) 或 None
        """
        # 获取最新的 tag 位置
        tag = self.get_fresh_tag(tag_id)
        if tag is None:
            rospy.logwarn("⚠️  无二维码数据")
            tag = self.latest_tags.get(tag_id)
            if tag is None:
                return None

        # x 用二维码推算（depth最不稳定，YOLO x不可信）
        # y/z 优先用 YOLO（横向和高度比固定偏移更准）
        gx = tag['x'] + GRASP_DEPTH_OFFSET
        _yolo = self.latest_yolo
        if _yolo is not None:
            _age = (rospy.Time.now() - _yolo['stamp']).to_sec()
            if _age < YOLO_VALID_DURATION:
                gy = float(_yolo['y'])
                gz = float(_yolo['z']) + 0.05  # 抬高10cm避开横梁
                rospy.loginfo(f"📍 融合: x={gx:.3f}(tag+{GRASP_DEPTH_OFFSET}) y={gy:.3f}(YOLO) z={gz:.3f}(YOLO+0.1m) age={_age:.2f}s")
                return (gx, gy, gz)
            else:
                rospy.logwarn(f"⚠️  YOLO数据过期 ({_age:.1f}s), y/z 用二维码推算")

        # 纯二维码推算（无YOLO备选）
        layer_offset_z = TAG_LAYER_OFFSET.get(tag_id, -0.3)
        gy = tag['y']
        gz = tag['z'] + layer_offset_z + 0.05  # 抬高10cm避开横梁
        rospy.logwarn(f"⚠️  纯二维码推算: x={gx:.3f}, y={gy:.3f}, z={gz:.3f}(+0.1m)")
        return (gx, gy, gz)

    # ==================== IK 请求构建 ====================

    def build_ik_request(self, x, y, z, use_left):
        """
        构建水平抓取 IK 请求
        参考: kuavo_sdk/scripts/arm_fk_ik/robot_arm_fk_ik.py
        
        优化要点：
        1. 使用宽松的 IK 参数提高求解成功率
        2. 根据目标位置动态计算初始猜测角度
        3. 调整目标姿态为更易求解的角度
        """
        zx, zy, zz = ROBOT_ZERO_52

        pose_cmd = twoArmHandPoseCmd()
        pose_cmd.ik_param = make_ik_solve_param(strict=True)  # 使用严格模式，提高求解精度
        pose_cmd.use_custom_ik_param = True
        pose_cmd.joint_angles_as_q0 = True
        
        # 根据目标位置计算更合理的初始猜测角度
        # 目标位置: x=前后距离, y=左右偏移, z=高度
        if use_left:
            # 左臂初始猜测：根据目标位置调整
            shoulder_pitch = -0.5  # 稍微抬起
            elbow_bend = -1.8  # 弯肘角度
            wrist_roll = 0.0
            pose_cmd.hand_poses.left_pose.joint_angles = np.array([
                0.35,      # shoulder_yaw (约20度)
                shoulder_pitch,  # shoulder_pitch
                0.0,       # shoulder_roll
                elbow_bend, # elbow_pitch
                0.0,       # elbow_roll
                wrist_roll, # wrist_pitch
                0.0        # wrist_yaw
            ])
            # 右臂保持零位附近
            pose_cmd.hand_poses.right_pose.joint_angles = np.array([0.35, 0.0, 0.0, -0.52, 0.0, 0.0, 0.0])
        else:
            # 右臂初始猜测
            shoulder_pitch = -0.5
            elbow_bend = -1.8
            wrist_roll = 0.0
            pose_cmd.hand_poses.right_pose.joint_angles = np.array([
                0.35,      # shoulder_yaw
                shoulder_pitch,  # shoulder_pitch
                0.0,       # shoulder_roll
                elbow_bend, # elbow_pitch
                0.0,       # elbow_roll
                wrist_roll, # wrist_pitch
                0.0        # wrist_yaw
            ])
            # 左臂保持零位附近
            pose_cmd.hand_poses.left_pose.joint_angles = np.array([0.35, 0.0, 0.0, -0.52, 0.0, 0.0, 0.0])

        # 计算目标姿态：面向目标的水平抓取姿态
        if use_left:
            rel_yaw = math.atan2(y - zy, x - zx) - 0.1
            quat = euler_to_quat(rel_yaw, -0.8, 0.0)  # 斜伸姿态，比垂直抓IK更易解
            pose_cmd.hand_poses.left_pose.pos_xyz      = np.array([x, y, z])  # 向下偏移0.05m
            pose_cmd.hand_poses.left_pose.quat_xyzw    = [quat.x, quat.y, quat.z, quat.w]
            pose_cmd.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3)
            
            pose_cmd.hand_poses.right_pose.pos_xyz      = np.array([0.15, -0.225, 0.1])
            pose_cmd.hand_poses.right_pose.quat_xyzw    = [0.0, 0.0, 0.0, 1.0]
            pose_cmd.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)
        else:
            rel_yaw = math.atan2(y - zy, x - zx) + 0.1
            quat = euler_to_quat(rel_yaw, -0.8, 0.0)  # 斜伸姿态，比垂直抓IK更易解
            pose_cmd.hand_poses.right_pose.pos_xyz      = np.array([x+0.0, y+0.15, z])  # x越大越靠右
            pose_cmd.hand_poses.right_pose.quat_xyzw    = [quat.x, quat.y, quat.z, quat.w]
            pose_cmd.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)
            
            pose_cmd.hand_poses.left_pose.pos_xyz      = np.array([0.15, 0.225, 0.1])
            pose_cmd.hand_poses.left_pose.quat_xyzw    = [0.0, 0.0, 0.0, 1.0]
            pose_cmd.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3)

        rospy.loginfo(f"📐 IK目标: x={x:.3f} y={y:.3f} z={z:.3f}, yaw={math.degrees(rel_yaw):.1f}°")
        return pose_cmd

    # ==================== 执行抓取 ====================

    def execute_grab(self, x, y, z):
        """
        完整水平抓取流程:
          1. 切换外部控制
          2. 张开手指
          3. IK 求解（多次求解取最优）
          4. 归零 → 抬手（防碰撞）
          5. 执行 IK 到达抓取位置
          6. 闭合手指
          7. 安全回收 → 归零
          8. 恢复自动摆臂
        """
        self.is_grabbing = True
        try:
            rospy.loginfo(f"🚀 水平抓取 base_link=({x:.3f}, {y:.3f}, {z:.3f})")

            # 确定使用左手还是右手
            use_left = y > 0
            rospy.loginfo(f"🖐 使用{'左' if use_left else '右'}手")

            # 1. 切换外部控制模式
            if not self.set_arm_control_mode(2):
                return False

            # 2. 张开夹爪
            rospy.loginfo("👐 张开夹爪")
            self.gripper.open_grippers()
            rospy.sleep(0.5)
            rospy.loginfo("✅ 夹爪已张开")

            # 3. IK 预检（多次求解取最优解）
            # 第一次抓取目标往下调0.03m，避免碰撞货架横梁
            grab_z = z - 0.01
            pose_cmd = self.build_ik_request(x, y, grab_z, use_left)
            best_res, best_err = self._solve_ik_multiple(pose_cmd, use_left, attempts=3)
            
            if best_res is None:
                rospy.logerr("❌ IK 求解失败, 放弃抓取")
                return False

            MAX_IK_ERROR = 0.03  # 3cm 阈值，误差超过则IK不可信
            if best_err > MAX_IK_ERROR:
                rospy.logerr(
                    f"❌ IK误差过大: {1e3*best_err:.1f}mm > {1e3*MAX_IK_ERROR:.0f}mm, "
                    f"位姿不可达，放弃抓取")
                return False
            rospy.loginfo(f"✅ IK 求解成功, 误差={1e3*best_err:.1f}mm")

            # 4. 归零
            rospy.loginfo("📐 手臂归零")
            self.publish_arm_target([1.5], ARM_HOME_DEG)
            if not self.wait_for_arm_arrival(ARM_HOME_DEG, min_wait=1.5, timeout=10.0):
                rospy.logwarn("⚠️  归零未到位，继续执行")

            # 5. 抬手动作（避免碰撞货架）
            rospy.loginfo("🙆 抬手（避免伸臂时碰撞货架）")
            lift_pre_pose = self._get_lift_pose(use_left)
            self.publish_arm_target([2.0], lift_pre_pose)
            self.wait_for_arm_arrival(lift_pre_pose, min_wait=2.0, timeout=5.0)

            # 6. 执行抓取（发布 IK 解）
            rospy.loginfo(f"🎯 执行抓取: x={x:.3f} y={y:.3f} z={grab_z:.3f}m")
            self._execute_ik_pose(best_res, use_left)

            # 6.5 手臂到位后，闭合夹爪前，让机器人向前走0.2m（避开货架横梁）
            rospy.loginfo("🚶 手臂到位，准备向前走0.2m...")
            forward_distance = 0.1  # 米
            forward_speed = 0.05  # m/s，走慢一点更稳
            forward_duration = forward_distance / forward_speed  # 4秒
            rospy.loginfo(f"   前进距离: {forward_distance}m, 速度: {forward_speed}m/s, 预计时间: {forward_duration:.1f}s")
            self.forward_goal(linear_x=forward_speed, duration=forward_duration)
            rospy.loginfo("   前进完成，等待1秒稳定...")
            rospy.sleep(1.0)

            # 7. 闭合夹爪
            rospy.loginfo("🤏 闭合夹爪")
            self.gripper.close_grippers()
            rospy.sleep(1.0)

            # 7.5 夹爪闭合后，向上抬臂0.2m（避免收回时碰撞货架）
            rospy.loginfo("📈 夹爪闭合，向上抬臂0.2m...")
            lift_up_z = z + 0.1  # 在当前抓取点上方0.2m
            rospy.loginfo(f"   目标位置: x={x:.3f} y={y:.3f} z={lift_up_z:.3f}m")
            
            # 构建向上抬臂的IK请求
            lift_pose_cmd = self.build_ik_request(x, y-0.09, lift_up_z, use_left)
            lift_res, lift_err = self._solve_ik_multiple(lift_pose_cmd, use_left, attempts=3)
            
            if lift_res is not None:
                rospy.loginfo(f"✅ 抬臂IK求解成功, 误差={1e3*lift_err:.1f}mm")
                self._execute_ik_pose(lift_res, use_left, limit_seventh_joint=False, limit_fifth_joint=False)  # 抬臂时不限制第5和第7个关节
                rospy.loginfo("✅ 抬臂到位")
            else:
                rospy.logwarn("⚠️  抬臂IK求解失败，跳过抬臂步骤")

            # 7.6 抬臂后，后退0.2m（夹爪保持闭合不变）
            rospy.loginfo("🚶 抬臂完成，准备后退0.2m...")
            backward_distance = 0.3  # 米
            backward_speed = 0.05  # m/s，慢速后退
            backward_duration = backward_distance / backward_speed  # 4秒
            rospy.loginfo(f"   后退距离: {backward_distance}m, 速度: {backward_speed}m/s, 预计时间: {backward_duration:.1f}s")
            self.forward_goal(linear_x=-backward_speed, duration=backward_duration)
            rospy.loginfo("   后退完成，夹爪保持闭合")

            # 8. 安全回收手臂（暂时跳过，保持当前姿态）
            # rospy.loginfo("🔄 安全回收手臂")
            # self._safe_recover(use_left)

            # 9. 归零（暂时跳过）
            # self.publish_arm_target([2.0], ARM_HOME_DEG)
            # self.wait_for_arm_arrival(ARM_HOME_DEG, timeout=5.0)

            # 10. 恢复姿态和控制模式（暂时跳过）
            # self.set_head(0, 0)
            # time.sleep(1.0)
            # self.set_arm_control_mode(1)
            rospy.loginfo("✅ 抓取流程完成")
            return True

        except Exception as e:
            rospy.logerr(f"抓取异常: {e}")
            import traceback; traceback.print_exc()
            return False
        finally:
            self.is_grabbing = False

    def _solve_ik_multiple(self, pose_cmd, use_left, attempts=10):
        """多次 IK 求解，选择误差最小的解"""
        best_res = None
        best_err = float('inf')
        target_pos = pose_cmd.hand_poses.left_pose.pos_xyz if use_left else pose_cmd.hand_poses.right_pose.pos_xyz
        
        rospy.loginfo(f"🔄 IK求解开始，目标位置: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        
        for attempt in range(attempts):
            rospy.loginfo(f"   尝试 {attempt+1}/{attempts}...")
            
            res = self.call_ik(pose_cmd)
            if res is not None:
                if res.success:
                    # 计算位置误差
                    actual_pos = res.hand_poses.left_pose.pos_xyz if use_left else res.hand_poses.right_pose.pos_xyz
                    err = np.linalg.norm(actual_pos - target_pos)
                    rospy.loginfo(f"   ✓ 成功! 误差={1e3*err:.1f}mm")
                    
                    if err < best_err:
                        best_err = err
                        best_res = res
                else:
                    rospy.loginfo(f"   ✗ 失败: {res.message if hasattr(res, 'message') else '未知原因'}")
            else:
                rospy.loginfo(f"   ✗ 服务调用失败")
            
            # 调整初始猜测角度（尝试不同的肘部弯曲角度）
            delta_angle = (attempt - attempts/2) * 0.2  # -0.4, -0.2, 0, +0.2, +0.4
            if use_left:
                pose_cmd.hand_poses.left_pose.joint_angles[3] = -1.8 + delta_angle  # elbow_pitch
                pose_cmd.hand_poses.left_pose.joint_angles[1] = -0.3 + delta_angle * 0.3  # shoulder_pitch
            else:
                pose_cmd.hand_poses.right_pose.joint_angles[3] = -1.8 + delta_angle  # elbow_pitch
                pose_cmd.hand_poses.right_pose.joint_angles[1] = -0.3 + delta_angle * 0.3  # shoulder_pitch
        
        if best_res is not None:
            rospy.loginfo(f"✅ IK求解完成，最优误差={1e3*best_err:.1f}mm")
        else:
            rospy.logerr("❌ 所有IK求解尝试均失败")
        
        return best_res, best_err

    def _get_lift_pose(self, use_left):
        """获取抬手姿态（避免碰撞货架）"""
        if use_left:
            return [
                30.0, 80.0, 20.0, -140.0, 20.0, 20.0, 0.0,
                20.0,  0.0, 0.0,  -30.0,  0.0,  0.0, 0.0
            ]
        else:
            return [
                20.0,  0.0, 0.0,  -30.0,  0.0,  0.0, 0.0,
                30.0, -80.0, -45.0, -140.0, 20.0, 20.0, 0.0
            ]

    def _execute_ik_pose(self, res, use_left, limit_seventh_joint=True, limit_fifth_joint=True):
        """执行 IK 求解结果"""
        MIN_THIRD_JOINT_DEG = 65.0  # 第三个关节(shoulder_roll)的最小角度限制
        min_third_joint_rad = math.radians(MIN_THIRD_JOINT_DEG)
        
        # 第5个关节(elbow_pitch)角度限制范围
        MIN_FIFTH_JOINT_DEG = 45.0
        MAX_FIFTH_JOINT_DEG = 50.0
        min_fifth_joint_rad = math.radians(MIN_FIFTH_JOINT_DEG)
        max_fifth_joint_rad = math.radians(MAX_FIFTH_JOINT_DEG)
        
        if use_left:
            left_joints = np.array(res.hand_poses.left_pose.joint_angles)
            left_joints[4] += 1.746  # elbow_pitch
            # left_joints[5] += 0.26  # shoulder_pitch
            
            # 限制第三个关节(shoulder_roll, 索引2)角度在45度以上
            if left_joints[1] < min_third_joint_rad:
                rospy.loginfo(f"⚠️  左臂第三个关节角度 {math.degrees(left_joints[1]):.1f}° < {MIN_THIRD_JOINT_DEG}°, 强制调整到 {MIN_THIRD_JOINT_DEG}°")
                left_joints[1] = min_third_joint_rad
            
            # 抓取时限制第5个关节(索引4)角度在45-55度之间
            if limit_fifth_joint:
                fifth_deg = math.degrees(left_joints[4])
                if left_joints[4] < min_fifth_joint_rad:
                    rospy.loginfo(f"⚠️  左臂第5个关节角度 {fifth_deg:.1f}° < {MIN_FIFTH_JOINT_DEG}°, 强制调整到 {MIN_FIFTH_JOINT_DEG}°")
                    left_joints[4] = min_fifth_joint_rad
                elif left_joints[4] > max_fifth_joint_rad:
                    rospy.loginfo(f"⚠️  左臂第5个关节角度 {fifth_deg:.1f}° > {MAX_FIFTH_JOINT_DEG}°, 强制调整到 {MAX_FIFTH_JOINT_DEG}°")
                    left_joints[4] = max_fifth_joint_rad
            
            # 抓取时限制第7个关节(索引6)角度在0度以上
            if limit_seventh_joint and left_joints[6] < 0.0:
                rospy.loginfo(f"⚠️  左臂第7个关节角度 {math.degrees(left_joints[6]):.1f}° < 0°, 强制调整到 0°")
                left_joints[6] = 0.0
            
            joints_rad = np.concatenate([left_joints, [0.35, 0.0, 0.0, -0.52, 0.0, 0.0, 0.0]])
        else:
            right_joints = np.array(res.hand_poses.right_pose.joint_angles)
            right_joints[4] += 1.24  # elbow_pitch
            right_joints[5] += 0.785  # shoulder_pitch
            right_joints[6] += -0.62  # elbow_pitch
            
            # 限制第三个关节(shoulder_roll, 索引2)角度在45度以上
            if right_joints[1] > -min_third_joint_rad:
                rospy.loginfo(f"⚠️  右臂第三个关节角度 {math.degrees(right_joints[1]):.1f}° < {MIN_THIRD_JOINT_DEG}°, 强制调整到 {MIN_THIRD_JOINT_DEG}°")
                right_joints[1] = -min_third_joint_rad
            
            # 抓取时限制第5个关节(索引4)角度在45-55度之间
            if limit_fifth_joint:
                fifth_deg = math.degrees(right_joints[4])
                if right_joints[4] < min_fifth_joint_rad:
                    rospy.loginfo(f"⚠️  右臂第5个关节角度 {fifth_deg:.1f}° < {MIN_FIFTH_JOINT_DEG}°, 强制调整到 {MIN_FIFTH_JOINT_DEG}°")
                    right_joints[4] = min_fifth_joint_rad
                elif right_joints[4] > max_fifth_joint_rad:
                    rospy.loginfo(f"⚠️  右臂第5个关节角度 {fifth_deg:.1f}° > {MAX_FIFTH_JOINT_DEG}°, 强制调整到 {MAX_FIFTH_JOINT_DEG}°")
                    right_joints[4] = max_fifth_joint_rad
            
            # 抓取时限制第7个关节(索引6)角度在0度以上
            if limit_seventh_joint and right_joints[6] < 0.0:
                rospy.loginfo(f"⚠️  右臂第7个关节角度 {math.degrees(right_joints[6]):.1f}° < 0°, 强制调整到 0°")
                right_joints[6] = 0.0
            
            joints_rad = np.concatenate([[0.35, 0.0, 0.0, -0.52, 0.0, 0.0, 0.0], right_joints])
        
        deg = [math.degrees(a) for a in joints_rad]
        
        rospy.loginfo(f"🤖 IK关节角(deg): {[f'{d:.1f}' for d in deg]}")
        rospy.loginfo(f"   关节角范围: min={min(deg):.1f}° max={max(deg):.1f}°")

        TRAJ_TIME   = 7.0   # 单次轨迹时长(s)，增加到7秒让手臂运动更慢
        MAX_WAIT    = 20.0 # 总超时(s)
        REPUB_INTV  = 2.0   # 每隔多少秒重新发布(防止控制器进入低增益保持)
        TOL_DEG     = 15.0  # 到位容差(度)

        rospy.loginfo("🤖 持续发送 IK 目标，等待到位（最多20s）...")
        self.publish_arm_target([TRAJ_TIME], deg)
        last_pub_t = rospy.Time.now()
        start_t    = rospy.Time.now()
        rate       = rospy.Rate(20)

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_t).to_sec()
            if elapsed > MAX_WAIT:
                rospy.logwarn("⚠️  抓取位置未完全到位，但继续执行")
                break

            # 周期性重发，防止控制器退出轨迹模式
            if (rospy.Time.now() - last_pub_t).to_sec() >= REPUB_INTV:
                self.publish_arm_target([TRAJ_TIME], deg)
                last_pub_t = rospy.Time.now()
                rospy.loginfo(f"🔄 重新发布 IK 目标 (elapsed={elapsed:.1f}s)")

            if self.current_sensors_data is not None:
                try:
                    arm_q = self.current_sensors_data.joint_data.joint_q[13:27]
                    max_err = max(abs(math.degrees(arm_q[i]) - deg[i]) for i in range(14))
                    rospy.loginfo_throttle(2.0, f"⏳ 手臂误差={max_err:.1f}° ({elapsed:.0f}s)")
                    if max_err <= TOL_DEG:
                        rospy.loginfo(f"✅ 手臂到位! 误差={max_err:.1f}°, 用时={elapsed:.1f}s")
                        break
                except Exception:
                    pass

            rate.sleep()

        rospy.loginfo("⏸ 到位，稳定等待 1.0s...")
        time.sleep(1.0)

    def _safe_recover(self, use_left):
        """安全回收手臂"""
        if use_left:
            # 分步回收
            recover_pose1 = [20.0, 90.0, 0.0, -90.0, 0.0, 0.0, 0.0,
                             20.0,  0.0, 0.0, -30.0, 0.0, 0.0, 0.0]
            self.publish_arm_target([2.0], recover_pose1)
            self.wait_for_arm_arrival(recover_pose1, timeout=5.0)
            
            recover_pose2 = [20.0, 90.0, 0.0, -30.0, 0.0, 0.0, 0.0,
                             20.0,  0.0, 0.0, -30.0, 0.0, 0.0, 0.0]
            self.publish_arm_target([2.0], recover_pose2)
            self.wait_for_arm_arrival(recover_pose2, timeout=5.0)
        else:
            recover_pose1 = [20.0,  0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
                             20.0, -90.0, 0.0, -90.0, 0.0, 0.0, 0.0]
            self.publish_arm_target([2.0], recover_pose1)
            self.wait_for_arm_arrival(recover_pose1, timeout=5.0)
            
            recover_pose2 = [20.0,  0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
                             20.0, -90.0, 0.0, -30.0, 0.0, 0.0, 0.0]
            self.publish_arm_target([2.0], recover_pose2)
            self.wait_for_arm_arrival(recover_pose2, timeout=5.0)

    # ==================== 主流程 ====================

    def run(self):
        rospy.sleep(1.0)
        init_pitch = self.head_pitch_deg if self.target_tag_id != 7 else 0.0
        self.set_head(0.0, init_pitch)

        if self.auto_grab:
            self._run_auto()
        else:
            rospy.loginfo("💤 等待模式 (auto=False), 设 _auto:=true 启动抓取")
            rospy.spin()

    def _run_auto(self):
        """完整自动抓取流程"""
        tag_id = self.target_tag_id
        layer = "上层" if tag_id == 7 else "下层"
        target_index = 1 if tag_id == 7 else 2

        # ---- Phase 1: 检测 AprilTag ----
        rospy.loginfo("=" * 55)
        rospy.loginfo(f"Phase 1: 检测 AprilTag {tag_id} ({layer})")
        rospy.loginfo("=" * 55)

        detect_pitch = self.head_pitch_deg if tag_id != 7 else 0.0
        self.set_head(0, detect_pitch)
        time.sleep(2.0)

        tag = self.wait_for_tag(tag_id, timeout=15.0)
        if tag is None:
            rospy.logerr(f"❌ 无法检测到 tag {tag_id}, 终止")
            return

        # ---- Phase 2: 导航到抓取距离 ----
        current_dist = tag['x']
        rospy.loginfo("=" * 55)
        rospy.loginfo(f"Phase 2: 导航 (当前距离 {current_dist:.3f}m → 目标 {self.nav_target_dist:.2f}m)")
        rospy.loginfo("=" * 55)

        if abs(current_dist - self.nav_target_dist) > NAV_TOLERANCE:
            before_dist = current_dist
            self.navigate_to_distance(tag_id, self.nav_target_dist)
            time.sleep(1.5)
            after_tag = self.get_fresh_tag(tag_id)
            after_dist = after_tag['x'] if after_tag else self.nav_target_dist
            rospy.loginfo(f"📏 靠近二维码: 前进距离 {before_dist - after_dist:.3f}m (前:{before_dist:.3f}m → 后:{after_dist:.3f}m)")
        else:
            rospy.loginfo(f"✅ 距离已合适 ({current_dist:.3f}m), 跳过导航")

        # 导航停止后等待机器人稳定，再刷新 YOLO 坐标（避免晃动导致坐标漂移）
        rospy.loginfo("⏸  等待机器人稳定 (1.0s)...")
        time.sleep(1.0)

        # ---- Phase 3: 获取抓取目标坐标 ----
        rospy.loginfo("=" * 55)
        rospy.loginfo(f"🔢 正在抓取: 目标{target_index} ({layer}, Tag {tag_id})")
        rospy.loginfo("=" * 55)

        target = self.get_grasp_target(tag_id)
        if target is None:
            rospy.logerr("❌ 无法获取抓取目标坐标, 终止")
            return
        
        gx, gy, gz = target

        rospy.loginfo(f"📍 抓取目标 (base_link):")
        rospy.loginfo(f"   x={gx:.3f}  y={gy:.3f}  z={gz:.3f}")

        # 安全检查
        if not (MIN_REACH_X <= gx <= MAX_REACH_X):
            rospy.logerr(f"❌ 抓取 x={gx:.3f} 超出臂长范围 [{MIN_REACH_X},{MAX_REACH_X}], 终止")
            return
        if not (MIN_HEIGHT_Z <= gz <= MAX_HEIGHT_Z):
            rospy.logerr(f"❌ 抓取 z={gz:.3f} 超出高度范围 [{MIN_HEIGHT_Z},{MAX_HEIGHT_Z}], 终止")
            return

        # 抬头避免手臂遮挡相机
        self.set_head(0, 0)
        time.sleep(0.5)

        # 执行抓取
        rospy.loginfo("=" * 55)
        rospy.loginfo(f"Phase 3: 抓取 ({layer})")
        rospy.loginfo(f"  grasp target: x={gx:.3f}  y={gy:.3f}  z={gz:.3f}")
        rospy.loginfo("=" * 55)

        self.execute_grab(gx, gy, gz)
        rospy.loginfo("🏁 自动抓取流程结束")


if __name__ == '__main__':
    try:
        GrabController().run()
    except rospy.ROSInterruptException:
        rospy.loginfo("抓取控制器已停止")
