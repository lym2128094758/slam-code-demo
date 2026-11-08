#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import time
import threading
import subprocess
import json
import os
import urllib.request
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

class AutoPatrol:
    def __init__(self):
        rospy.init_node('auto_patrol_node', anonymous=True)
        
        # 创建一个连接到 move_base 的 Action 客户端
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        rospy.loginfo("正在等待 move_base 导航服务器启动...")
        self.client.wait_for_server()
        rospy.loginfo("导航服务器已连接！准备开始自动巡检...")

        self.clear_costmaps_thread = None
        self.clear_costmaps_running = False
        self.clear_costmaps_count = 0
        
        # 发布导航完成信号
        self.completed_pub = rospy.Publisher('/auto_patrol/completed', Bool, queue_size=10)
# ==========================================
        # 📍 你的巡检路线路书
        # 格式：[ (x, y, orientation_z, orientation_w), ... ]
        # ==========================================
        self.waypoints = [
            (0.688395118713379, 0.0264607667922974, 0.020130658369109773, 0.9997973577648753),
            (9.585846214294434, -0.3756355929374695, -0.6427876096865393, 0.766044443118978)
            # (9.205846214294434, 1.5956355929374695, -0.6984228243479552, 0.7156853767053127),
            # (9.405846214294434, 2.5956355929374695, -0.6984228243479552, 0.7156853767053127),
            # (9.755846214294434, 6.0956355929374695, -0.6984228243479552, 0.7156853767053127),
            # (9.9, 7.67377, 1.0, 0.0),
        ]

    def _debug_report(self, hypothesis_id, message, data=None, location="auto_patrol.py"):
        env_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../.dbg/nav-obstacle-collision.env")
        )
        server_url = "http://127.0.0.1:7777/event"
        session_id = "nav-obstacle-collision"
        try:
            with open(env_path, "r", encoding="utf-8") as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        server_url = line.split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1]
        except Exception:
            pass

        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {message}",
            "data": data or {},
            "ts": int(time.time() * 1000),
        }
        try:
            req = urllib.request.Request(
                server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=0.2).read()
        except Exception:
            pass

    def start_clear_costmaps_thread(self):
        """当前版本不启动清理代价地图线程。"""
        return

    def stop_clear_costmaps_thread(self):
        """当前版本不停止清理代价地图线程。"""
        return

    def wait_for_position_reached(self, target_x, target_y, tolerance=0.5, timeout=120):
        rospy.loginfo(f"等待到达位置附近: ({target_x}, {target_y}) 容差: {tolerance}m")
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn(f"⚠️ 等待位置到达超时，继续下一目标点")
                return False
            
            try:
                qpos = rospy.wait_for_message('/mujoco/qpos', JointState, timeout=2.0)
                robot_x = qpos.position[0]
                robot_y = qpos.position[1]
                
                dist = ((robot_x - target_x)**2 + (robot_y - target_y)**2)**0.5
                if dist < tolerance:
                    rospy.loginfo(f"✅ 位置已到达附近！距离目标: {dist:.2f}m")
                    return True
            except rospy.ROSException:
                pass
            
            rate.sleep()
        return False

    def send_goal(self, point, point_name, check_orientation=True):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        goal.target_pose.pose.position.z = 0.0
        
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = point[2]
        goal.target_pose.pose.orientation.w = point[3]

        rospy.loginfo(f"🚀 正在前往目标点: {point_name}")
        # #region debug-point A:send-goal
        self._debug_report(
            "A",
            "move_base goal sent",
            {
                "point_name": point_name,
                "goal_x": point[0],
                "goal_y": point[1],
                "goal_qz": point[2],
                "goal_qw": point[3],
                "check_orientation": check_orientation,
            },
            "auto_patrol.py:send_goal",
        )
        # #endregion
        self.client.send_goal(goal)

        if check_orientation:
            wait = self.client.wait_for_result(rospy.Duration(300))
            if not wait:
                rospy.logwarn(f"⚠️ 前往 {point_name} 超时，继续前往下一个点")
                # #region debug-point A:goal-timeout
                self._debug_report(
                    "A",
                    "move_base goal timeout",
                    {"point_name": point_name, "state": self.client.get_state()},
                    "auto_patrol.py:send_goal",
                )
                # #endregion
                self.client.cancel_goal()
                return True
            else:
                rospy.loginfo(f"✅ 成功到达 {point_name}！")
                # #region debug-point A:goal-result
                self._debug_report(
                    "A",
                    "move_base goal finished",
                    {"point_name": point_name, "state": self.client.get_state()},
                    "auto_patrol.py:send_goal",
                )
                # #endregion
                return True
        else:
            wait = self.client.wait_for_result(rospy.Duration(300))
            if not wait:
                rospy.logwarn(f"⚠️ 前往 {point_name} 超时，继续前往下一个点")
                # #region debug-point A:goal-timeout-no-orientation
                self._debug_report(
                    "A",
                    "move_base goal timeout without orientation check",
                    {"point_name": point_name, "state": self.client.get_state()},
                    "auto_patrol.py:send_goal",
                )
                # #endregion
                self.client.cancel_goal()
                return True
            else:
                rospy.loginfo(f"✅ 成功到达 {point_name}！")
                # #region debug-point A:goal-result-no-orientation
                self._debug_report(
                    "A",
                    "move_base goal finished without orientation check",
                    {"point_name": point_name, "state": self.client.get_state()},
                    "auto_patrol.py:send_goal",
                )
                # #endregion
                return True

    def start_patrol(self):
        # 可以在出发前等几秒，或者等待之前写的"开机环视"转完
        rospy.sleep(3.0) 

        # 已按需求禁用巡航过程中的代价地图清理逻辑
        # self.start_clear_costmaps_thread()

        for i, point in enumerate(self.waypoints):
            # 第一个航点不检查朝向，只检查位置
            if i == 0:
                success = self.send_goal(point, f"航点_{i+1}", check_orientation=False)
            else:
                success = self.send_goal(point, f"航点_{i+1}")
            if not success:
                rospy.logwarn("巡检任务被迫中断。")
                break
            
            # 到达一个点后，休息 2 秒再前往下一个点
            rospy.sleep(2.0)

        # 已按需求禁用巡航过程中的代价地图清理逻辑
        # self.stop_clear_costmaps_thread()
        rospy.loginfo("🎉 所有巡检路线已全部跑完！")
        
        # 发布导航完成信号
        rospy.sleep(0.5)
        self.completed_pub.publish(Bool(data=True))
        rospy.loginfo("已发布导航完成信号")

    def switch_to_leju_walk_controller(self):
        rospy.loginfo("切换到 leju_walk_controller...")
        try:
            cmd = "rosservice call /humanoid_controller/switch_controller \"controller_name: 'leju_walk_controller'\""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            rospy.loginfo(f"切换控制器响应: {result.stdout.strip()}")
            if result.returncode == 0:
                rospy.loginfo("控制器切换成功！")
            else:
                rospy.logwarn(f"控制器切换失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            rospy.logerr("切换控制器超时")
        except Exception as e:
            rospy.logerr(f"切换控制器失败: {e}")

    def publish_velocity(self, linear_x):
        rospy.loginfo(f"发布速度: linear_x={linear_x}")
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rate = rospy.Rate(10)
        vel = Twist()
        vel.linear.x = linear_x
        vel.linear.y = 0.0
        vel.linear.z = 0.0
        vel.angular.x = 0.0
        vel.angular.y = 0.0
        vel.angular.z = 0.0

        for _ in range(50):
            pub.publish(vel)
            rate.sleep()

if __name__ == '__main__':
    try:
        patrol = AutoPatrol()
        patrol.start_patrol()
    except rospy.ROSInterruptException:
        pass
