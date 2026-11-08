#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零件抓取控制器 - 基于 YOLO 视觉检测 + IK 逆解

功能:
    1. 订阅 /yolo_parts/detections 获取零件在 base_link 下的坐标
    2. 手臂归零
    3. 抬手（避免碰桌子）
    4. IK 逆解并发布轨迹到达抓取位置
    5. 到位后停顿2秒，闭合夹爪
    6. 抬手（与初始抬手轨迹一致）
    7. IK 逆解到收纳箱上方
    8. 打开夹爪，将物品放到收纳箱

使用方法:
    rosrun craic_task_template grab_parts.py

订阅:
    /yolo_parts/detections — 零件检测结果（base_link坐标系）

控制接口:
    /arm_traj_change_mode         — 手臂控制模式
    /ik/two_arm_hand_pose_cmd_srv — IK 逆解服务
    /kuavo_arm_target_poses       — 关节角目标
    /control_robot_hand_position  — 夹爪控制
"""

import sys
import os
import math
import time
import threading
import subprocess
import signal
import numpy as np
import rospy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../craic_simulator/utils'))
from gripper_controller import GripperController

from geometry_msgs.msg import PointStamped
from kuavo_msgs.msg import (
    armTargetPoses,
    robotHandPosition,
    twoArmHandPoseCmd,
    ikSolveParam,
    sensorsData,
)
from kuavo_msgs.srv import (
    changeArmCtrlMode,
    changeArmCtrlModeRequest,
    twoArmHandPoseCmdSrv,
    fkSrv,
)


# ============================ 常量 / 配置 ============================

CLOSE_HAND = 0.8
OPEN_HAND = 0.0

ROBOT_ZERO_52 = (-0.012, -0.225, -0.315)

ARM_HOME_DEG = [
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
]
box = [
    -100.0, 30.0, -60.0, 0.0, 0.0, 0.0, 0.0,
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
]

LIFT_LEFT_DEG = [
    20.0, 90.0, 30.0, -60.0, -50.0, 20.0, 0.0,
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
]

LIFT_TEICE_LEFT_DEG = [
    0.0, 0.0, 0.0, -120.0, -115.0, -50.0, 0.0,
    20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0,
]
MIN_REACH_X = 0.15
MAX_REACH_X = 0.70
MIN_HEIGHT_Z = -0.50
MAX_HEIGHT_Z = 0.70

TRAJ_TIME = 5.0
GRASP_PAUSE_TIME = 2.0
LIFT_OFFSET = 0.2

GRAB_DISTANCE_THRESHOLD = 0.08
MAX_WHITE_PARTS = 3

# ============================ 抓取姿态与偏置参数 (可微调) ============================
GRASP_PITCH = -0.8   # -0.8 弧度（约 -45°），斜伸姿态比垂直抓IK更易解


# ============================ 工具函数 ============================

class Q:
    __slots__ = ('w', 'x', 'y', 'z')
    def __init__(self): self.w = self.x = self.y = self.z = 0.0


def euler_to_quat(yaw, pitch, roll=0.0):
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
    p = ikSolveParam()
    if strict:
        p.major_optimality_tol   = 1e-3
        p.major_feasibility_tol  = 1e-3
        p.minor_feasibility_tol  = 1e-3
        p.major_iterations_limit = 1000
        p.oritation_constraint_tol = 0.08
        p.pos_constraint_tol     = 1e-3
        p.pos_cost_weight        = 0.5
    else:
        p.major_optimality_tol   = 1e-2
        p.major_feasibility_tol  = 1e-2
        p.minor_feasibility_tol  = 1e-2
        p.major_iterations_limit = 500
        p.oritation_constraint_tol = 0.3
        p.pos_constraint_tol     = 5e-3
        p.pos_cost_weight        = 0.7
    return p


# ============================ 抓取控制器 ============================

class GrabPartsController:
    def __init__(self):
        rospy.init_node('grab_parts_controller', anonymous=True)

        self.latest_detection = None
        self.detected_parts = {}  # 缓存所有检测到的零件
        self.current_sensors_data = None
        self.is_grabbing = False
        self.hold_gripper = False
        self.is_first_grab = True  # 标记是否为程序启动后的第一次抓取

        self.left_box_pos = (0.705, 0.501, 0.458)
        self.right_box_pos = (0.705, -0.4554, 0.451)

        self.white_grab_count = 0
        self.grabbed_positions = []
        self.white_grab_done = False
        self.no_white_start_time = None  # 记录视野中消失白色零件的时间点，用于稳定性判定

        self.tf_listener = None

        self.head_pub = rospy.Publisher('/robot_head_motion_data', robotHandPosition, queue_size=10)
        self.arm_pub = rospy.Publisher('/kuavo_arm_target_poses', armTargetPoses, queue_size=10)
        self.hand_pub = rospy.Publisher('/control_robot_hand_position', robotHandPosition, queue_size=10)

        self.gripper = GripperController()

        # 注册退出钩子，确保子进程被关闭
        rospy.on_shutdown(self._cleanup)

        # 检查是否已有检测节点在运行，如果有，尝试先关闭它，防止名称冲突
        rospy.loginfo("🔍 检查是否存在冲突的 yolo_parts_detector 节点...")
        os.system("rosnode kill /yolo_parts_detector > /dev/null 2>&1")
        rospy.sleep(1.0) # 等待节点退出完成

        yolo_parts_launch = os.path.join(
            os.path.dirname(__file__),
            '../../yolov8_detection/launch/yolo_parts.launch'
        )
        yolo_parts_launch = os.path.abspath(yolo_parts_launch)
        rospy.loginfo(f"🚀 启动 yolo_parts.launch: {yolo_parts_launch}")
        
        # 使用 start_new_session 确保整个进程组可以被一起关闭
        self.yolo_process = subprocess.Popen(
            ['roslaunch', 'yolov8_detection', 'yolo_parts.launch'],
            preexec_fn=os.setsid
        )
        rospy.sleep(3.0)

        rospy.Subscriber('/yolo_parts/detections', PointStamped, self._detection_cb, queue_size=1)
        rospy.Subscriber('/sensors_data_raw', sensorsData, self._sensors_cb, queue_size=1)

        rospy.loginfo("🎯 grab_parts 控制器已启动")
        rospy.loginfo("📡 等待检测到零件...")

    def _cleanup(self):
        """退出时的清理逻辑"""
        rospy.loginfo("🧹 正在清理资源并关闭子进程...")
        if hasattr(self, 'yolo_process') and self.yolo_process:
            try:
                # 关闭整个进程组（包括 roslaunch 启动的所有子节点）
                os.killpg(os.getpgid(self.yolo_process.pid), signal.SIGTERM)
                self.yolo_process.wait(timeout=2.0)
                rospy.loginfo("✅ yolo_parts.launch 已关闭")
            except Exception as e:
                rospy.logwarn(f"⚠️ 关闭子进程时出错: {e}")
        
        # 确保夹爪控制停止
        self.hold_gripper = False
        self.is_grabbing = False

    def _detection_cb(self, msg):
        try:
            name = msg.header.frame_id.lower()
            x = float(msg.point.x)
            y = float(msg.point.y)
            z = float(msg.point.z)

            # 屏蔽通过视觉更新收纳箱位置，使用固定位置
            if "box" in name:
                return

            if "part" not in name:
                return

            det_info = {
                'name': msg.header.frame_id,
                'x': x,
                'y': y,
                'z': z,
                'stamp': rospy.Time.now()
            }
            
            # 使用名称作为 key 存储最新的检测结果
            self.detected_parts[msg.header.frame_id] = det_info
            self.latest_detection = det_info
            
            rospy.loginfo_throttle(5.0, f"📦 缓存零件: {det_info['name']} "
                                         f"({det_info['x']:.3f}, "
                                         f"{det_info['y']:.3f}, "
                                         f"{det_info['z']:.3f}) | "
                                         f"包含'white': {'white' in name}")
        except Exception as e:
            rospy.logwarn(f"检测回调异常: {e}")

    def _sensors_cb(self, msg):
        self.current_sensors_data = msg

    def set_arm_control_mode(self, mode):
        try:
            rospy.wait_for_service('/arm_traj_change_mode', timeout=5.0)
            cli = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            req = changeArmCtrlModeRequest()
            req.control_mode = mode
            resp = cli(req)
            if resp.result:
                rospy.loginfo(f"✅ arm mode → {mode}")
                return True
            rospy.logwarn(f"⚠️  arm mode 失败: {resp.message}")
            return False
        except Exception as e:
            rospy.logerr(f"❌ arm mode 服务异常: {e}")
            return False

    def call_ik(self, pose_cmd):
        try:
            rospy.wait_for_service('/ik/two_arm_hand_pose_cmd_srv', timeout=5.0)
            srv = rospy.ServiceProxy('/ik/two_arm_hand_pose_cmd_srv', twoArmHandPoseCmdSrv)
            return srv(pose_cmd)
        except Exception as e:
            rospy.logerr(f"❌ IK 异常: {e}")
            return None

    def call_fk(self, joint_angles):
        try:
            rospy.wait_for_service('/ik/fk_srv', timeout=2.0)
            srv = rospy.ServiceProxy('/ik/fk_srv', fkSrv)
            # fk_srv takes a list of 14 floats
            return srv(joint_angles).hand_poses
        except Exception as e:
            rospy.logerr(f"❌ FK 异常: {e}")
            return None

    def _hold_gripper_closed(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown() and self.is_grabbing and self.hold_gripper:
            self.gripper.control_left_gripper(255.0)
            rate.sleep()

    def publish_arm_target(self, times, values_deg):
        # 强制右手始终保持固定角度 (20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0)
        fixed_right_deg = [20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]
        if len(values_deg) == 14:
            values_deg = list(values_deg[:7]) + fixed_right_deg

        msg = armTargetPoses()
        msg.times = list(times)
        msg.values = list(values_deg)
        self.arm_pub.publish(msg)

    def wait_for_arm_arrival(self, target_joints_deg, min_wait=0, timeout=15.0, tolerance=15.0):
        # rospy.loginfo(f"⏳ 等待手臂到位 (超时={timeout}s, 容差={tolerance}°)")
        start_time = rospy.Time.now()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                rospy.logwarn(f"⚠️  手臂到位超时 ({timeout}s)")
                return False

            if self.current_sensors_data is None:
                rate.sleep()
                continue

            try:
                arm_q = self.current_sensors_data.joint_data.joint_q[13:27]
                max_error = max(
                    abs(math.degrees(arm_q[i]) - target_joints_deg[i])
                    for i in range(14)
                )
                rospy.loginfo_throttle(2.0, f"⏳ 手臂误差={max_error:.2f}° (已等 {elapsed:.1f}s)")

                if max_error <= tolerance:
                    if elapsed < min_wait:
                        rate.sleep()
                        continue
                    rospy.loginfo(f"✅ 手臂到位! 误差={max_error:.2f}°, 用时={elapsed:.1f}s")
                    return True
            except Exception:
                pass

            rate.sleep()
        return False

    def print_current_fk_feedback(self, label="目标到位反馈"):
        """获取并打印当前实测的手臂关节角和末端 FK 位置"""
        if self.current_sensors_data is None:
            rospy.logwarn(f"⚠️ {label}: 无法获取传感器数据，跳过反馈")
            return

        try:
            current_joints_rad = list(self.current_sensors_data.joint_data.joint_q[13:27])
            current_joints_deg = [math.degrees(q) for q in current_joints_rad]
            
            rospy.loginfo(f"---------------- {label} ----------------")
            # 仅打印左手实测关节角度 (0-6)
            rospy.loginfo(f"📊 [实测关节角度-左] {['%.2f' % d for d in current_joints_deg[0:7]]}")

            fk_res = self.call_fk(current_joints_rad)
            if fk_res:
                l_pos = fk_res.left_pose.pos_xyz
                # 关节 6 和 7 的索引是 5,6
                l6, l7 = current_joints_deg[5], current_joints_deg[6]
                
                rospy.loginfo(f"📍 [当前实测FK] 左手末端: ({l_pos[0]:.3f}, {l_pos[1]:.3f}, {l_pos[2]:.3f}), 旋转(j6,j7): {l6:.2f}°, {l7:.2f}°")
            rospy.loginfo(f"--------------------------------------------")
        except Exception as e:
            rospy.logwarn(f"⚠️ {label}: 反馈获取失败: {e}")

    def build_ik_request(self, x, y, z, fixed_z=None):
        # 直接使用检测到的坐标
        target_x = x 
        target_y = y 
        target_z = fixed_z if fixed_z is not None else z

        pose_cmd = twoArmHandPoseCmd()
        pose_cmd.ik_param = make_ik_solve_param(strict=False)
        pose_cmd.use_custom_ik_param = True
        pose_cmd.joint_angles_as_q0 = True

        # 左手初始猜测
        pose_cmd.hand_poses.left_pose.joint_angles = np.array([
            0.35, -0.5, 0.0, -1.8, 0.0, 0.0, 0.0
        ])
        # 右手固定角度 (20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0)
        pose_cmd.hand_poses.right_pose.joint_angles = np.array([
            math.radians(20.0), 0.0, 0.0, math.radians(-30.0), 0.0, 0.0, 0.0
        ])

        rel_yaw = math.radians(-15.0)  # 强制修改为 -15 度  
        quat = euler_to_quat(rel_yaw, GRASP_PITCH, 0.0)
        
        # 左手目标位姿
        pose_cmd.hand_poses.left_pose.pos_xyz = np.array([target_x, target_y, target_z])
        pose_cmd.hand_poses.left_pose.quat_xyzw = [quat.x, quat.y, quat.z, quat.w]
        pose_cmd.hand_poses.left_pose.elbow_pos_xyz = np.zeros(3)
        
        # 右手目标位姿设为对应固定角度的末端位置（或者给一个安全点，IK会自动解出接近猜测值的解）
        # 这里给一个固定的参考位姿，避免 IK 引擎为了凑右手而影响左手
        pose_cmd.hand_poses.right_pose.pos_xyz = np.array([0.15, -0.225, 0.1])
        pose_cmd.hand_poses.right_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        pose_cmd.hand_poses.right_pose.elbow_pos_xyz = np.zeros(3)

        rospy.loginfo(f"📐 IK目标: x={target_x:.3f} y={target_y:.3f} z={target_z:.3f}, yaw={math.degrees(rel_yaw):.1f}°")
        return pose_cmd

    def solve_ik(self, pose_cmd, attempts=20):
        best_res = None
        best_err = float('inf')
        target_pos = pose_cmd.hand_poses.left_pose.pos_xyz
        rospy.loginfo(f"🎯 IK 目标位置: x={target_pos[0]:.3f} y={target_pos[1]:.3f} z={target_pos[2]:.3f}")

        # 根据目标高度自适应生成初始关节猜测，确保能伸直臂够到低矮桌面（如 z = -0.068 米）
        if target_pos[2] < 0.05:
            # 低目标（桌面零件）：需要手臂充分向下伸直，肘部折弯（joint 3）设在 [-0.3, -1.2] 区间
            shoulder_roll_values = [-0.2, -0.4, -0.6]
            elbow_bend_values = [-0.3, -0.6, -0.9, -1.2]
        else:
            # 高目标：手臂可以适度折弯
            shoulder_roll_values = [-0.3, -0.5, -0.7, -0.9]
            elbow_bend_values = [-1.4, -1.6, -1.8, -2.0]

        for attempt in range(attempts):
            res = self.call_ik(pose_cmd)
            if res is not None:
                if res.success:
                    # 使用 FK 验证 IK 结果
                    raw_joints_rad = list(res.hand_poses.left_pose.joint_angles) + list(res.hand_poses.right_pose.joint_angles)
                    fk_res = self.call_fk(raw_joints_rad)
                    
                    if fk_res:
                        actual_pos = fk_res.left_pose.pos_xyz
                    else:
                        actual_pos = res.hand_poses.left_pose.pos_xyz
                    
                    err = np.linalg.norm(actual_pos - target_pos)
                    
                    # 检查第四个关节（索引3）是否小于-10度（-0.174弧度）
                    joint4 = res.hand_poses.left_pose.joint_angles[3]
                    joint4_deg = math.degrees(joint4)
                    
                    valid = joint4 < -0.174
                    
                    if err < 0.05:
                        status = "✓" if valid else "✗"
                        rospy.loginfo(f"   尝试 {attempt+1}/{attempts}: {status} FK误差={1e3*err:.1f}mm, 关节4={joint4_deg:.1f}°, FK实际=({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})")
                    
                    if valid and err < best_err:
                        best_err = err
                        best_res = res
                        if err < 0.01:
                            break
                else:
                    if attempt < 3:
                        rospy.loginfo(f"   尝试 {attempt+1}/{attempts}: ✗ 失败")
            else:
                if attempt < 3:
                    rospy.loginfo(f"   尝试 {attempt+1}/{attempts}: ✗ 服务调用失败")

            # 网格网化计算：使用 idx_roll 与 idx_elbow 依次组合
            idx_roll = attempt % len(shoulder_roll_values)
            idx_elbow = (attempt // len(shoulder_roll_values)) % len(elbow_bend_values)

            # 自适应前伸肩俯仰（joint 0）：低目标零件时，微幅调高肩俯仰初始角度（0.45 弧度 ≒ 25.8°）协助手臂下够
            target_pitch = 0.45 if target_pos[2] < 0.05 else 0.35

            pose_cmd.hand_poses.left_pose.joint_angles[0] = target_pitch
            pose_cmd.hand_poses.left_pose.joint_angles[1] = shoulder_roll_values[idx_roll]
            pose_cmd.hand_poses.left_pose.joint_angles[3] = elbow_bend_values[idx_elbow]

        MAX_IK_ERROR = 0.05
        if best_res is not None:
            joint4 = best_res.hand_poses.left_pose.joint_angles[3]
            rospy.loginfo(f"✅ IK求解完成，最优误差={1e3*best_err:.1f}mm, 关节4={math.degrees(joint4):.1f}°")
            if best_err > MAX_IK_ERROR:
                rospy.logwarn(f"⚠️  IK误差过大 ({1e3*best_err:.1f}mm > {1e3*MAX_IK_ERROR:.0f}mm)，目标可能不可达")
        else:
            rospy.logerr("❌ 所有IK求解尝试均失败（可能找不到满足关节4 < -20°的解）")

        return best_res, best_err

    def execute_ik_pose(self, res):
        # -----------------------------

        left_joints = np.array(res.hand_poses.left_pose.joint_angles)
        left_joints[4] -= 1.57  # 发布前减90度
        
        # 右手始终保持固定角度 (20.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0)
        right_joints_rad = [math.radians(20.0), 0.0, 0.0, math.radians(-30.0), 0.0, 0.0, 0.0]
        joints_rad = np.concatenate([left_joints, right_joints_rad])

        deg = [math.degrees(a) for a in joints_rad]
        rospy.loginfo(f"🤖 IK关节角: {[f'{d:.1f}' for d in deg]}")

        rospy.loginfo("🤖 发送 IK 目标...")
        self.publish_arm_target([TRAJ_TIME], deg)
        
        start_t = rospy.Time.now()
        while (rospy.Time.now() - start_t).to_sec() < TRAJ_TIME + 2:
            if self.wait_for_arm_arrival(deg, timeout=10.0, tolerance=5.0):
                break
            self.publish_arm_target([TRAJ_TIME], deg)
            rospy.sleep(0.5)

    def execute_grab(self, x, y, z, part_name):
        self.is_grabbing = True
        try:
            rospy.loginfo(f"🚀 开始抓取 {part_name} base_link=({x:.3f}, {y:.3f}, {z:.3f})")

            rospy.loginfo(f"🖐 仅使用左手执行任务")

            if not self.set_arm_control_mode(2):
                return False

            rospy.loginfo("👐 张开左夹爪")
            self.gripper.control_left_gripper(0.0)
            rospy.sleep(0.5)

            if self.is_first_grab:
                rospy.loginfo("� 手臂归零 (仅首次抓取)")
                self.publish_arm_target([1.5], ARM_HOME_DEG)
                self.wait_for_arm_arrival(ARM_HOME_DEG, min_wait=1.5, timeout=10.0)

                rospy.loginfo("�🙆 首次抓取：执行 LIFT_LEFT_DEG -> LIFT_TEICE_LEFT_DEG 序列")
                self.publish_arm_target([2.0], LIFT_LEFT_DEG)
                self.wait_for_arm_arrival(LIFT_LEFT_DEG, min_wait=2.0, timeout=10.0)

                self.publish_arm_target([2.0], LIFT_TEICE_LEFT_DEG)
                self.wait_for_arm_arrival(LIFT_TEICE_LEFT_DEG, min_wait=2.0, timeout=10.0)
                self.is_first_grab = False
            else:
                # 移除归零和 LIFT_LEFT_DEG 中间过程，直接确保在 LIFT_TEICE_LEFT_DEG 姿态
                rospy.loginfo("�🙆 非首次抓取：直接确保到位 LIFT_TEICE_LEFT_DEG")
                self.publish_arm_target([2.0], LIFT_TEICE_LEFT_DEG)
                self.wait_for_arm_arrival(LIFT_TEICE_LEFT_DEG, min_wait=2.0, timeout=10.0)

            # 在 IK 逆解之前输出一下现在抓的是白色的还是黑色的
            color_str = "⚪ 白色" if "white" in part_name.lower() else "⚫ 黑色"
            rospy.loginfo(f"🔍 [IK求解前] 当前准备抓取的零件颜色: {color_str}")

            # 每次抓取的目标 z 固定为 -0.105；补偿只作用在 x/y
            ideal_x = x
            ideal_y = y 
            ideal_z = -0.105
            fixed_grab_z = -0.105
            
            cx, cy = 0.0, 0.0
            max_attempts = 2
            best_res = None
            
            for attempt in range(max_attempts):
                comp_x = x + cx
                comp_y = y + cy
                
                pose_cmd = self.build_ik_request(comp_x, comp_y, fixed_grab_z, fixed_z=fixed_grab_z)
                res, err = self.solve_ik(pose_cmd)
                
                if res is None:
                    rospy.logerr("❌ IK 求解失败")
                    if attempt == 0:
                        return False
                    else:
                        break
                
                best_res = res
                rospy.loginfo(f"✅ [动态补偿 第{attempt+1}次] IK 成功, 误差={1e3*err:.1f}mm")
                
                rospy.loginfo(f"🎯 执行抓取...")
                self.execute_ik_pose(best_res)
                
                # 等待手臂物理惯性完全消除并稳定
                rospy.loginfo("⏸ 等待 1.5 秒让手臂完全到位并稳定...")
                rospy.sleep(1.5)
                
                # --- FK 误差检测与反馈补偿 ---
                try:
                    current_joints_rad = list(self.current_sensors_data.joint_data.joint_q[13:27])
                    fk_res = self.call_fk(current_joints_rad)
                    if fk_res:
                        actual_pos = fk_res.left_pose.pos_xyz
                        dx = ideal_x - actual_pos[0]
                        dy = ideal_y - actual_pos[1]
                        dz = ideal_z - actual_pos[2]
                        
                        err_x, err_y, err_z = abs(dx), abs(dy), abs(dz)
                        rospy.loginfo(f"📏 [误差检测] 理想目标: ({ideal_x:.3f}, {ideal_y:.3f}, {ideal_z:.3f})")
                        rospy.loginfo(f"📏 [误差检测] 实测FK: ({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})")
                        rospy.loginfo(f"📏 [误差检测] 差值(理想-实测): dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
                        
                        # 精度要求提高到 8mm 以内
                        if err_x <= 0.020 and err_y <= 0.020 and err_z <= 0.015:
                            rospy.loginfo("✅ 当前实测位姿满足误差要求，停止修正。")
                            break
                        else:
                            if attempt < max_attempts - 1:
                                rospy.loginfo(f"⚠️ 误差未达标(X:{1e3*dx:.1f}mm, Y:{1e3*dy:.1f}mm, Z:{1e3*dz:.1f}mm)，下一轮只按本轮 x/y 误差覆盖补偿，z 保持固定...")
                                
                                # --- 避障抬手逻辑 ---
                                rospy.loginfo("⬆️ [避障保护] 在修正移动前，先发布固定安全轨迹抬高避障，防止夹爪刮蹭...")
                                self.publish_arm_target([2.0], LIFT_TEICE_LEFT_DEG)
                                self.wait_for_arm_arrival(LIFT_TEICE_LEFT_DEG, min_wait=1.0, timeout=5.0)

                                # 本轮 x/y 误差是多少，下一轮就补偿多少；z 不参与补偿
                                cx = dx
                                cy = dy
                                rospy.loginfo(f"🔄 补偿后偏移量: cx={cx:.3f}, cy={cy:.3f}, 固定抓取z={fixed_grab_z:.3f}")
                            else:
                                rospy.logwarn("⚠️ 达到最大修正次数，强行继续执行。")
                    else:
                        rospy.logwarn("⚠️ 无法获取 FK 结果，跳过反馈修正。")
                        break
                except Exception as e:
                    rospy.logwarn(f"⚠️ 反馈修正异常: {e}")
                    break

            if best_res is None:
                return False

            rospy.loginfo(f"⏸ 到位后停顿 {GRASP_PAUSE_TIME} 秒...")
            time.sleep(GRASP_PAUSE_TIME)

            # 在闭合夹爪前输出 FK 反馈
            self.print_current_fk_feedback(label="抓取位实测反馈")

            rospy.loginfo("🤏 闭合左夹爪")
            self.gripper.control_left_gripper(255.0)
            rospy.loginfo("⏸ 闭合后等待 2 秒...")
            time.sleep(2.0)

            rospy.loginfo("📈 直接抬手至 LIFT_TEICE_LEFT_DEG...")
            self.publish_arm_target([2.0], LIFT_TEICE_LEFT_DEG)
            self.wait_for_arm_arrival(LIFT_TEICE_LEFT_DEG, min_wait=2.0, timeout=10.0)
            
            rospy.loginfo("✅ 抬臂完成")

            # 开启后台高频闭合保持信号
            self.hold_gripper = True
            rospy.loginfo("🔒 持续保持夹爪闭合状态")
            gripper_hold_thread = threading.Thread(target=self._hold_gripper_closed)
            gripper_hold_thread.daemon = True
            gripper_hold_thread.start()

            part_name_lower = part_name.lower()
            rospy.loginfo(f"🔍 分拣判断 - 零件名称: {part_name} | 小写: {part_name_lower} | 包含'white': {'white' in part_name_lower}")
            
            if "white" in part_name_lower:
                target_box = self.left_box_pos
                box_side = "左边"
                rospy.loginfo(f"⚪ 判定为白色零件 → 放置左边")
            else:
                target_box = self.right_box_pos
                box_side = "右边"
                rospy.loginfo(f"⚫ 判定为黑色零件 → 放置右边")

            if target_box is None:
                rospy.logerr(f"❌ 未检测到{box_side}收纳箱，无法放置物品")
                return False

            box_x, box_y, box_z = target_box
            # z 轴高度已经在固定坐标中指定，直接使用
            rospy.loginfo(f"📦 {part_name} → {box_side}收纳箱上方: ({box_x:.3f}, {box_y:.3f}, {box_z:.3f})")
            storage_pose_cmd = self.build_ik_request(box_x, box_y, box_z)
            storage_res, storage_err = self.solve_ik(storage_pose_cmd)

            if storage_res is not None:
                rospy.loginfo(f"✅ 收纳箱IK求解成功, 误差={1e3*storage_err:.1f}mm")
                self.execute_ik_pose(storage_res)
            else:
                rospy.logwarn("⚠️  收纳箱IK求解失败，使用固定轨迹放置")
                self.publish_arm_target([TRAJ_TIME], box)
                self.wait_for_arm_arrival(box, min_wait=2.0, timeout=10.0)

            rospy.loginfo(f"⏸ 到位后停顿 {GRASP_PAUSE_TIME} 秒...")
            time.sleep(GRASP_PAUSE_TIME)
            
            # 在放下物品前输出 FK 反馈
            self.print_current_fk_feedback(label="放置位实测反馈")

            # 必须先关闭后台高频闭合控制，否则 control_left_gripper 的瞬间会被后台再次强行 close_grippers
            self.hold_gripper = False
            rospy.loginfo("👐 打开左夹爪，放下物品")
            self.gripper.control_left_gripper(0.0)
            time.sleep(1.0)
            rospy.loginfo(f"✅ {part_name} 已放到{box_side}收纳箱")

            rospy.loginfo("🙆 直接抬手至 LIFT_TEICE_LEFT_DEG（跳过归零）...")
            self.publish_arm_target([2.0], LIFT_TEICE_LEFT_DEG)
            self.wait_for_arm_arrival(LIFT_TEICE_LEFT_DEG, min_wait=2.0, timeout=10.0)

            rospy.loginfo("✅ 抓取流程完成")
            return True

        except Exception as e:
            rospy.logerr(f"抓取异常: {e}")
            import traceback; traceback.print_exc()
            return False
        finally:
            self.hold_gripper = False
            self.is_grabbing = False

    def run(self):
        rospy.loginfo("💤 等待检测到零件 (策略：优先抓取所有白色零件，清空后再抓黑色)...")
        while not rospy.is_shutdown():
            if not self.is_grabbing:
                # 1. 清理超过 3 秒的旧检测数据
                now = rospy.Time.now()
                self.detected_parts = {k: v for k, v in self.detected_parts.items() if (now - v['stamp']).to_sec() < 3.0}

                if not self.detected_parts:
                    self.no_white_start_time = None # 没有任何零件时重置计时
                    rospy.sleep(0.1)
                    continue

                # 2. 筛选白色和黑色零件
                white_parts = [v for k, v in self.detected_parts.items() if "white" in k.lower()]
                black_parts = [v for k, v in self.detected_parts.items() if "black" in k.lower()]

                target_det = None
                
                if white_parts:
                    # 优先选择白色零件
                    target_det = white_parts[0]
                    self.no_white_start_time = None  # 发现白色零件，重置计时
                    rospy.loginfo(f"⚪ 发现白色零件，优先抓取: {target_det['name']}")
                elif black_parts:
                    # 如果没有白色零件，但有黑色零件
                    if self.no_white_start_time is None:
                        self.no_white_start_time = now
                        rospy.loginfo("⏳ 视野中未见白色零件，开始计时判定（需持续 2.0s 以确认切换到黑色）...")
                    
                    wait_duration = (now - self.no_white_start_time).to_sec()
                    if wait_duration >= 2.0:
                        target_det = black_parts[0]
                        rospy.loginfo(f"⚫ 确认无白色零件（已等待 {wait_duration:.1f}s），开始抓取黑色: {target_det['name']}")
                    else:
                        rospy.loginfo_throttle(0.5, f"⏳ 正在确认白色零件是否清空... 已等待 {wait_duration:.1f}s")
                        rospy.sleep(0.1)
                        continue

                if target_det:
                    x, y, z = target_det['x'], target_det['y'], target_det['z']
                    part_name = target_det['name']

                    if not (MIN_REACH_X <= x <= MAX_REACH_X):
                        rospy.logwarn(f"⚠️  x={x:.3f} 超出范围 [{MIN_REACH_X},{MAX_REACH_X}]")
                        # 移除该目标，避免死循环
                        self.detected_parts.pop(part_name, None)
                    elif not (MIN_HEIGHT_Z <= z <= MAX_HEIGHT_Z):
                        rospy.logwarn(f"⚠️  z={z:.3f} 超出范围 [{MIN_HEIGHT_Z},{MAX_HEIGHT_Z}]")
                        self.detected_parts.pop(part_name, None)
                    else:
                        rospy.loginfo(f"🎯 启动抓取: {part_name}, 位置=({x:.3f}, {y:.3f}, {z:.3f})")
                        success = self.execute_grab(x, y, z, part_name)
                        if success:
                            if "white" in part_name.lower():
                                self.white_grab_count += 1
                                rospy.loginfo(f"✅ 成功归类白色物品进度: {self.white_grab_count}")
                            # 抓取成功后从缓存中移除，防止重复抓取同一位置（虽然视觉会更新，但移除更保险）
                            self.detected_parts.pop(part_name, None)

            rospy.sleep(0.1)


if __name__ == '__main__':
    try:
        GrabPartsController().run()
    except rospy.ROSInterruptException:
        rospy.loginfo("抓取控制器已停止")
