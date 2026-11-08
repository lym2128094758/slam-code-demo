#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import time
import threading
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Empty

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
# ==========================================
        # 📍 你的巡检路线路书
        # 格式：[ (x, y, orientation_z, orientation_w), ... ]
        # ==========================================
        self.waypoints = [
            # 这是你刚刚手动踩出来的精准目标点
            (0.788395118713379, 1.0264607667922974, 0.020130658369109773, 0.9997973577648753),
            (9.305846214294434, 0.5956355929374695, 0.6984228243479552, 0.7156853767053127),
            #(0.9170210361480713, 1.1562472581863403, -0.005659938372528657, 0.9999839824205282)
            # --- 以下是备用点，已按你的要求注释掉 ---
            # (2.0, 1.5, 0.0, 1.0),     # 备用点 A
            # (5.0, 1.5, 0.707, 0.707), # 备用点 B
            # (5.0, 4.0, 1.0, 0.0)      # 备用点 C
        ]

    def start_clear_costmaps_thread(self):
        def clear_loop():
            while self.clear_costmaps_running:
                try:
                    rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
                    clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
                    clear_costmaps()
                    rospy.loginfo("已清理代价地图")
                except rospy.ServiceException as e:
                    rospy.logwarn(f"清理代价地图失败: {e}")
                except rospy.ROSException:
                    pass
                time.sleep(1.5)

        self.clear_costmaps_running = True
        self.clear_costmaps_thread = threading.Thread(target=clear_loop)
        self.clear_costmaps_thread.daemon = True
        self.clear_costmaps_thread.start()

    def stop_clear_costmaps_thread(self):
        self.clear_costmaps_running = False
        if self.clear_costmaps_thread:
            self.clear_costmaps_thread.join(timeout=2.0)

    def send_goal(self, point, point_name, check_orientation=True):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # 填入坐标
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        goal.target_pose.pose.position.z = 0.0
        
        # 填入姿态（四元数）
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = point[2]
        goal.target_pose.pose.orientation.w = point[3]

        rospy.loginfo(f"🚀 正在前往目标点: {point_name}")
        self.client.send_goal(goal)

        if check_orientation:
            # 完整检查：等待结果，包含朝向
            wait = self.client.wait_for_result(rospy.Duration(120))
            if not wait:
                rospy.logerr(f"❌ 前往 {point_name} 超时或失败！")
                self.client.cancel_goal()
                return False
            else:
                rospy.loginfo(f"✅ 成功到达 {point_name}！")
                return True
        else:
            # 只检查位置：等待8秒后认为已到达位置
            rospy.sleep(8.0)
            self.client.cancel_goal()
            rospy.loginfo(f"✅ 位置已到达 {point_name}（不检查朝向）！")
            return True

    def start_patrol(self):
        # 可以在出发前等几秒，或者等待之前写的"开机环视"转完
        rospy.sleep(3.0) 

        # 启动清理代价地图线程
        self.start_clear_costmaps_thread()

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

        # 停止清理代价地图线程
        self.stop_clear_costmaps_thread()
        rospy.loginfo("🎉 所有巡检路线已全部跑完！")

if __name__ == '__main__':
    try:
        patrol = AutoPatrol()
        patrol.start_patrol()
    except rospy.ROSInterruptException:
        pass