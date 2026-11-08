#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import math

def spin_robot():
    # 初始化节点
    rospy.init_node('init_spin_node', anonymous=True)
    
    # 创建一个发布者，往底盘发送速度指令
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    # 1. 缓冲期：等待 3 秒，让 Mujoco 仿真、AMCL 和代价地图完全启动
    rospy.loginfo("等待系统初始化...")
    rospy.sleep(3.0)

    # 2. 开始旋转
    rospy.loginfo("系统就绪，开始 360 度环视扫描...")
    vel_msg = Twist()
    
    # ⚠️ 极度关键：夸父是双足机器人，转太快会被自己的惯性甩倒！
    # 0.4 rad/s 是一个非常优雅且求稳的测试速度
    vel_msg.angular.z = 0.4 

    # 计算转一圈（2π 弧度）需要多长时间
    spin_duration = (0.5 * math.pi) / vel_msg.angular.z
    
    t0 = rospy.Time.now().to_sec()
    
    # 在规定时间内持续发送旋转指令
    while not rospy.is_shutdown():
        t1 = rospy.Time.now().to_sec()
        if (t1 - t0) > spin_duration:
            break
        #pub.publish(vel_msg)
        rate.sleep()

    # 3. 完美刹车
    vel_msg.angular.z = 0.0
    pub.publish(vel_msg)
    rospy.loginfo("环视扫描完毕，AMCL 已收敛，Costmap 已点亮！准备接收导航目标！")

if __name__ == '__main__':
    try:
        spin_robot()
    except rospy.ROSInterruptException:
        pass