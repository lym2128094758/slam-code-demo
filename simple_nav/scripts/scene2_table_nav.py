#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf.transformations as tr
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float64MultiArray


class Scene2TableNavigator:
    def __init__(self):
        rospy.init_node('scene2_table_nav_node', anonymous=True)

        self.target_x = float(rospy.get_param('~target_x', -0.05017119149487522))
        self.target_y = float(rospy.get_param('~target_y', 0.3726238066941296))
        self.target_yaw_deg = float(rospy.get_param('~target_yaw_deg', 90.0))
        self.xy_tolerance = float(rospy.get_param('~xy_tolerance', 0.03))
        self.yaw_tolerance_rad = math.radians(float(rospy.get_param('~yaw_tolerance_deg', 1.0)))
        self.timeout = float(rospy.get_param('~timeout', 60.0))
        self.max_vel = float(rospy.get_param('~max_vel', 0.15))
        self.max_angular_vel = float(rospy.get_param('~max_angular_vel', 0.3))
        self.kp = float(rospy.get_param('~kp', 0.8))
        self.kp_yaw = float(rospy.get_param('~kp_yaw', 0.5))

        self.current_yaw = None
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.completed_pub = rospy.Publisher('/scene2_nav/completed', Bool, queue_size=10)
        self.imu_sub = rospy.Subscriber('/imu_data', Imu, self.imu_callback)

    def imu_callback(self, msg):
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        _, _, self.current_yaw = tr.euler_from_quaternion(quaternion)

    def get_robot_pose_from_qpos(self):
        qpos = rospy.wait_for_message('/mujoco/qpos', Float64MultiArray, timeout=2.0)
        robot_x = qpos.data[0]
        robot_y = qpos.data[1]
        robot_yaw = 0.0

        if len(qpos.data) >= 7:
            qw = qpos.data[3]
            qx = qpos.data[4]
            qy = qpos.data[5]
            qz = qpos.data[6]
            _, _, robot_yaw = tr.euler_from_quaternion([qx, qy, qz, qw])

        return robot_x, robot_y, robot_yaw

    def stop_robot(self, repeat=10):
        stop_cmd = Twist()
        for _ in range(repeat):
            self.cmd_vel_pub.publish(stop_cmd)
            rospy.sleep(0.05)

    def clip(self, value):
        return max(-self.max_vel, min(self.max_vel, value))

    def clip_angular(self, value):
        return max(-self.max_angular_vel, min(self.max_angular_vel, value))

    def rotate_to_target(self, target_deg, tolerance_deg=1.0):
        target_rad = math.radians(target_deg)
        tolerance_rad = math.radians(tolerance_deg)
        rospy.loginfo("开始转向对准 %.1f 度 (误差允许 %.1f 度)...", target_deg, tolerance_deg)
        rate = rospy.Rate(15)
        
        while not rospy.is_shutdown():
            if self.current_yaw is None:
                rospy.logwarn_throttle(2.0, "等待 IMU 数据以进行转向对准...")
                rate.sleep()
                continue
                
            yaw_error = target_rad - self.current_yaw
            # 归一化到 [-pi, pi]
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            while yaw_error < -math.pi: yaw_error += 2 * math.pi
            
            if abs(yaw_error) <= tolerance_rad:
                self.stop_robot()
                rospy.loginfo("转向对准完成! 当前角度: %.2f 度", math.degrees(self.current_yaw))
                break
                
            cmd = Twist()
            angular_z = self.kp_yaw * yaw_error
            # 增加最小角速度防止死区
            min_angular_vel = 0.05
            if abs(angular_z) < min_angular_vel:
                angular_z = min_angular_vel if yaw_error > 0 else -min_angular_vel
                
            cmd.angular.z = self.clip_angular(angular_z)
            self.cmd_vel_pub.publish(cmd)
            
            rospy.loginfo_throttle(1.0, "转向中... 目标: %.1f, 当前: %.2f, 误差: %.2f", 
                                   target_deg, math.degrees(self.current_yaw), math.degrees(yaw_error))
            rate.sleep()

    def run(self):
        rospy.sleep(1.0)
        rospy.loginfo(
            "开始第二关任务导航流程: 旋转对准 -> XY导航 -> 精准旋转"
        )

        # 阶段 0: 初始转向对准 (先大致转过去)
        self.rotate_to_target(self.target_yaw_deg, tolerance_deg=5.0)

        start_time = rospy.Time.now()
        rate = rospy.Rate(15)

        # 阶段 1: XY 平面导航
        reached_xy = False
        rospy.loginfo("进入阶段 1: 导航到目标点 (%.3f, %.3f)", self.target_x, self.target_y)
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > self.timeout:
                rospy.logerr("qpos 导航超时，未到达目标点")
                self.stop_robot()
                self.completed_pub.publish(Bool(data=False))
                return

            try:
                robot_x, robot_y, robot_yaw = self.get_robot_pose_from_qpos()
            except rospy.ROSException:
                rate.sleep()
                continue

            error_x_world = self.target_x - robot_x
            error_y_world = self.target_y - robot_y
            distance = math.hypot(error_x_world, error_y_world)

            if distance <= self.xy_tolerance:
                self.stop_robot()
                rospy.loginfo("位置导航到位，进入阶段 2: 精准角度对准...")
                reached_xy = True
                break

            # 将世界坐标误差投影到机器人本体坐标系，再发布速度
            cmd_x = math.cos(robot_yaw) * error_x_world + math.sin(robot_yaw) * error_y_world
            cmd_y = -math.sin(robot_yaw) * error_x_world + math.cos(robot_yaw) * error_y_world

            cmd = Twist()
            cmd.linear.x = self.clip(self.kp * cmd_x)
            cmd.linear.y = self.clip(self.kp * cmd_y)
            # 移动过程中也尝试维持目标角度
            if self.current_yaw is not None:
                yaw_error = math.radians(self.target_yaw_deg) - self.current_yaw
                while yaw_error > math.pi: yaw_error -= 2 * math.pi
                while yaw_error < -math.pi: yaw_error += 2 * math.pi
                cmd.angular.z = self.clip_angular(self.kp_yaw * yaw_error)
            else:
                cmd.angular.z = 0.0
                
            self.cmd_vel_pub.publish(cmd)

            rospy.loginfo_throttle(
                1.0,
                "qpos 导航中: 距离目标=%.3f, 当前角度=%.2f",
                distance,
                math.degrees(self.current_yaw) if self.current_yaw is not None else 0.0
            )
            rate.sleep()
            
        # 阶段 2: 最终精准角度对准
        if reached_xy:
            self.rotate_to_target(self.target_yaw_deg, tolerance_deg=1.0)
            rospy.loginfo("第二关导航到位并精准对准完毕!")
            self.completed_pub.publish(Bool(data=True))
        else:
            self.completed_pub.publish(Bool(data=False))


if __name__ == '__main__':
    try:
        Scene2TableNavigator().run()
    except rospy.ROSInterruptException:
        pass
