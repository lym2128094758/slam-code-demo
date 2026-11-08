#!/usr/bin/env python3
"""
场景一：安全巡检任务脚本

功能：
1. 启动导航
2. 自动完成目标点巡航
3. 导航结束后关闭 move_base
4. 切换 MPC 控制器
5. 先对准0度方向（对正斜坡）
6. 使用 MPC 步态控制连续发送速度上坡到 x=1.54-1.55 位置
7. 再次精确对准0度
8. 等待3秒后启动 stairClimbPlanner.py
9. 等待32秒后发送 x=0.2 的速度走2秒
10. 停止
"""

import argparse
import sys
import os
import subprocess
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../craic_simulator/utils'))
from sim_launcher import SimLauncher


def main():

    parser = argparse.ArgumentParser(description="场景一：安全巡检")
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    args = parser.parse_args()

    # ========================================
    # 启动仿真
    # ========================================
    launcher = SimLauncher(scene="scene1", seed=args.seed)
    launcher.start(node_name="scene1_patrol")

    import rospy

    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, Float64MultiArray

    rospy.loginfo("=== 场景一：安全巡检启动 ===")

    # ========================================
    # Publisher
    # ========================================
    cmd_vel_pub = rospy.Publisher(
        '/cmd_vel',
        Twist,
        queue_size=10
    )

    rospy.sleep(2.0)

    # ========================================
    # 启动导航
    # ========================================
    rospy.loginfo("启动导航...")

    nav_process = subprocess.Popen(
        ["roslaunch", "simple_nav", "my_navigation.launch"],
        preexec_fn=os.setsid
    )

    rospy.sleep(5.0)

    # ========================================
    # 等待导航完成
    # ========================================
    nav_completed = False

    def nav_callback(msg):
        nonlocal nav_completed

        if msg.data:
            nav_completed = True
            rospy.loginfo("收到导航完成信号")

    rospy.Subscriber(
        '/auto_patrol/completed',
        Bool,
        nav_callback
    )

    rospy.loginfo("等待导航完成...")

    timeout = 300

    start_wait = rospy.Time.now()

    while not rospy.is_shutdown():

        if nav_completed:
            rospy.loginfo("✅ 导航完成")
            break

        elapsed = (rospy.Time.now() - start_wait).to_sec()

        if elapsed > timeout:
            rospy.logwarn("⚠️ 导航等待超时")
            break

        rospy.sleep(1.0)

    # ========================================
    # 关闭 move_base
    # ========================================
    rospy.loginfo("关闭 move_base...")

    try:
        subprocess.run(
            ["rosnode", "kill", "/move_base"],
            capture_output=True
        )

        rospy.loginfo("✅ move_base 已关闭")

    except Exception as e:
        rospy.logwarn(f"关闭 move_base 失败: {e}")

    rospy.sleep(2.0)

    # ========================================
    # 切换 MPC 控制器
    # ========================================
    rospy.loginfo("切换 MPC 控制器...")

    try:

        from kuavo_msgs.srv import switchController

        rospy.wait_for_service(
            '/humanoid_controller/switch_controller',
            timeout=5.0
        )

        switch_controller = rospy.ServiceProxy(
            '/humanoid_controller/switch_controller',
            switchController
        )

        switch_controller('mpc')

        rospy.loginfo("✅ 已切换 MPC 控制器")

    except Exception as e:

        rospy.logwarn(f"MPC控制器切换失败: {e}")
        rospy.logwarn("继续使用当前步态控制器")

    rospy.sleep(2.0)

    # ========================================
    # 订阅 amcl_pose 以获取地图坐标
    # ========================================
    from geometry_msgs.msg import PoseWithCovarianceStamped
    current_map_x = None
    current_map_y = None

    def amcl_pose_callback(msg):
        nonlocal current_map_x, current_map_y
        current_map_x = msg.pose.pose.position.x
        current_map_y = msg.pose.pose.position.y

    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, amcl_pose_callback)

    # ========================================
    # 第一步：对准90度方向
    # ========================================
    rospy.loginfo("=== 第一步：对准90度方向 ===")

    import tf
    import math
    from sensor_msgs.msg import Imu

    current_yaw = None

    def imu_callback(msg):
        nonlocal current_yaw
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        current_yaw = euler[2]

    imu_sub = rospy.Subscriber('/imu_data', Imu, imu_callback)

    while current_yaw is None and not rospy.is_shutdown():
        rospy.loginfo("等待IMU数据...")
        rospy.sleep(0.5)

    rate = rospy.Rate(20)
    stop_cmd = Twist()
    
    target_yaw_first = 89 * math.pi / 180  # 90度

    # 粗略对准
    rospy.loginfo("=== 粗略对准 ===")
    rough_yaw_tolerance = 0.15
    rough_angular_speed = 0.3

    rospy.loginfo(f"当前偏航角: {math.degrees(current_yaw):.1f}度")

    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw_first - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"粗略对准中... 当前偏航角: {math.degrees(current_yaw):.1f}度, 误差: {math.degrees(error):.1f}度")

        if abs(error) < rough_yaw_tolerance:
            rospy.loginfo(f"✅ 粗略对准完成！当前偏航角: {math.degrees(current_yaw):.1f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = rough_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(0.5)

    # 微调精确对准（误差控制在0.5度以内）
    rospy.loginfo("=== 微调精确对准 ===")
    fine_yaw_tolerance = 0.00873  # 0.5度 = 0.5 * π / 180 ≈ 0.00873弧度
    fine_angular_speed = 0.05

    rospy.loginfo(f"当前偏航角: {math.degrees(current_yaw):.1f}度")

    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw_first - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"微调对准中... 当前偏航角: {math.degrees(current_yaw):.2f}度, 误差: {math.degrees(error):.2f}度")

        if abs(error) < fine_yaw_tolerance:
            rospy.loginfo(f"✅ 精确对准完成！当前偏航角: {math.degrees(current_yaw):.2f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = fine_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(1.0)

    rospy.loginfo("✅ 已对准90度方向，准备后退")

    # ========================================
    # 第二步：发布向后速度，后退至 y=7.5 停止
    # ========================================
    rospy.loginfo("=== 第二步：向后移动至 y=7.5 ===")

    cmd = Twist()
    cmd.linear.x = -0.13
    cmd.linear.y = 0.0
    cmd.linear.z = 0.0
    cmd.angular.x = 0.0
    cmd.angular.y = 0.0
    cmd.angular.z = 0.0

    rospy.loginfo("持续发送向后 /cmd_vel ...")

    target_y = 6.3

    while not rospy.is_shutdown():
        if current_map_y is None:
            rospy.logwarn_throttle(2.0, "等待 amcl_pose 坐标数据...")
            rate.sleep()
            continue

        # 因为对准了 90度（y轴正方向），向后退（x为负）会使地图上的 y 坐标增大
        # 我们需要后退直到 y 坐标到达 7.5
        
        # 持续发布向后速度
        cmd_vel_pub.publish(cmd)
        rospy.loginfo_throttle(1.0, f"后退中... 当前地图坐标: x={current_map_x:.3f}m, y={current_map_y:.3f}m, speed=-0.15m/s")
        
        # 由于向后走，y坐标会越来越大。如果 y >= 7.5，则停止
        if current_map_y >= target_y:
            rospy.loginfo(f"✅ 到达目标位置！y={current_map_y:.3f}m >= {target_y}m")
            # 立即发送停止命令
            for _ in range(50):
                cmd_vel_pub.publish(stop_cmd)
            break

        rate.sleep()

    rospy.loginfo("✅ 第二步后退完成")
    rospy.sleep(1.0)

    # ========================================
    # 插入：对准0度方向并向前走
    # ========================================
    rospy.loginfo("=== 对准0度方向并向前走 ===")
    target_yaw_zero = 0.0

    # 粗略对准0度
    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw_zero - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"0度粗略对准中... 当前偏航角: {math.degrees(current_yaw):.1f}度, 误差: {math.degrees(error):.1f}度")

        if abs(error) < rough_yaw_tolerance:
            rospy.loginfo(f"✅ 0度粗略对准完成！当前偏航角: {math.degrees(current_yaw):.1f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = rough_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(0.5)

    # 微调精确对准0度
    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw_zero - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"0度微调对准中... 当前偏航角: {math.degrees(current_yaw):.2f}度, 误差: {math.degrees(error):.2f}度")

        if abs(error) < fine_yaw_tolerance:
            rospy.loginfo(f"✅ 0度精确对准完成！当前偏航角: {math.degrees(current_yaw):.2f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = fine_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(1.0)

    # 这里的目标是为了走到 1.624-1.626 那个斜坡前的位置
    rospy.loginfo("=== 开始以0.08速度前进 ===")
    cmd_forward = Twist()
    cmd_forward.linear.x = 0.08
    
    target_x_min = 1.624
    target_x_max = 1.626
    arrived = False

    while not rospy.is_shutdown():
        try:
            qpos = rospy.wait_for_message('/mujoco/qpos', Float64MultiArray, timeout=2.0)
            robot_x = qpos.data[0]

            if robot_x < target_x_min:
                cmd_vel_pub.publish(cmd_forward)
                rospy.loginfo_throttle(1.0, f"前进中... x={robot_x:.3f}m, speed=0.08m/s")
            elif robot_x >= target_x_min and robot_x <= target_x_max:
                if not arrived:
                    rospy.loginfo(f"✅ 到达目标位置！x={robot_x:.3f}m 在 [{target_x_min:.3f}, {target_x_max:.3f}] 范围内")
                    arrived = True
                for _ in range(50):
                    cmd_vel_pub.publish(stop_cmd)
                break
            elif robot_x > target_x_max:
                cmd_forward.linear.x = -0.04
                rospy.loginfo_throttle(1.0, f"⚠️ 超过目标位置！后退中... x={robot_x:.3f}m, speed=-0.04m/s")
                cmd_vel_pub.publish(cmd_forward)

        except rospy.ROSException:
            rospy.logwarn("获取位置失败")

        rate.sleep()

    rospy.loginfo("✅ 到达目标位置完成")
    rospy.sleep(1.0)

    # ========================================
    # 第三步：再次精确对准0度
    # ========================================
    rospy.loginfo("=== 第三步：再次精确对准0度 ===")

    rospy.loginfo(f"当前偏航角: {math.degrees(current_yaw):.1f}度")

    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = -current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"再次对准中... 当前偏航角: {math.degrees(current_yaw):.2f}度, 误差: {math.degrees(error):.2f}度")

        if abs(error) < fine_yaw_tolerance:
            rospy.loginfo(f"✅ 再次精确对准完成！当前偏航角: {math.degrees(current_yaw):.2f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = fine_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    for _ in range(50):
        cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(1.5)

    # ========================================
    # 等待3秒后启动 stairClimbPlanner.py
    # ========================================
    rospy.loginfo("=== 等待3秒后启动上下楼梯脚本 ===")
    rospy.sleep(3.0)

    rospy.loginfo("=== 启动 stairClimbPlanner.py ===")
    stair_planner_path = os.path.join(os.path.dirname(__file__), 'stairClimbPlanner.py')

    stair_process = subprocess.Popen(
        ['python3', stair_planner_path],
        preexec_fn=os.setsid
    )

    rospy.loginfo("✅ stairClimbPlanner.py 已启动")

    # ========================================
    # 等待32秒
    # ========================================
    rospy.loginfo("=== 等待32秒 ===")
    rospy.sleep(32.0)

    rospy.loginfo("✅ 32秒等待完成")

    # ========================================
    # 发送 x=0.2 的速度走4秒
    # ========================================
    rospy.loginfo("=== 发送 x=0.2 的速度走4秒 ===")

    forward_cmd = Twist()
    forward_cmd.linear.x = 0.2
    forward_cmd.linear.y = 0.0
    forward_cmd.linear.z = 0.0
    forward_cmd.angular.x = 0.0
    forward_cmd.angular.y = 0.0
    forward_cmd.angular.z = 0.0

    start_forward = rospy.Time.now()

    while not rospy.is_shutdown():
        elapsed = (rospy.Time.now() - start_forward).to_sec()

        if elapsed > 4.0:
            break

        cmd_vel_pub.publish(forward_cmd)
        rospy.loginfo_throttle(0.5, f"前进中... elapsed={elapsed:.1f}s, speed=0.2m/s")
        rate.sleep()

    # 立即发送停止命令
    for _ in range(50):
        cmd_vel_pub.publish(stop_cmd)
    rospy.loginfo("✅ 前进4秒完成")

    # ========================================
    # 第四步：停顿2秒并180度右转
    # ========================================
    rospy.loginfo("=== 第四步：停顿2秒并180度右转 ===")
    rospy.sleep(2.0)

    rospy.loginfo("开始180度右转...")
    target_yaw = -math.pi  # 180度，使用 -pi 确保右转方向
    
    # 粗略对准
    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"180度粗略右转中... 当前偏航角: {math.degrees(current_yaw):.1f}度, 误差: {math.degrees(error):.1f}度")

        if abs(error) < rough_yaw_tolerance:
            rospy.loginfo(f"✅ 180度粗略对准完成！当前偏航角: {math.degrees(current_yaw):.1f}度")
            break

        rotate_cmd = Twist()
        # 强制使用右转速度 -0.3，直到接近目标
        rotate_cmd.angular.z = -rough_angular_speed
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(1.5)

    # 微调精确对准
    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"180度微调对准中... 当前偏航角: {math.degrees(current_yaw):.2f}度, 误差: {math.degrees(error):.2f}度")

        if abs(error) < fine_yaw_tolerance:
            rospy.loginfo(f"✅ 180度精确对准完成！当前偏航角: {math.degrees(current_yaw):.2f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = fine_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(1.0)

    # ========================================
    # 新增：在移动到第二次上楼梯点前，先对准 y 坐标
    # ========================================
    rospy.loginfo("=== 对准 y=-3.50 坐标 ===")
    target_y = -3.50
    y_tolerance = 0.1
    y_speed = 0.1

    while not rospy.is_shutdown():
        try:
            qpos = rospy.wait_for_message('/mujoco/qpos', Float64MultiArray, timeout=2.0)
            robot_y = qpos.data[1]
            
            y_error = robot_y - target_y
            
            if abs(y_error) <= y_tolerance:
                rospy.loginfo(f"✅ y 坐标对准完成！当前 y={robot_y:.3f}m, 误差={y_error:.3f}m")
                for _ in range(50):
                    cmd_vel_pub.publish(stop_cmd)
                break
                
            y_cmd = Twist()
            if y_error > y_tolerance:
                # y 坐标偏大，需要向 y 轴负方向移动
                # 由于机器人现在面向 180 度（-x 方向），机器人的左侧是 -y，右侧是 +y
                # 要向真实的 -y 移动，在机器人坐标系下需要向左平移，即 linear.y > 0
                y_cmd.linear.y = y_speed
                rospy.loginfo_throttle(0.5, f"y坐标对准中 (向-y移动)... 当前y={robot_y:.3f}m, 目标={target_y}")
            else:
                # y 坐标偏小，需要向 y 轴正方向移动
                # 在机器人坐标系下面向 180 度时，需要向右平移，即 linear.y < 0
                y_cmd.linear.y = -y_speed
                rospy.loginfo_throttle(0.5, f"y坐标对准中 (向+y移动)... 当前y={robot_y:.3f}m, 目标={target_y}")
                
            cmd_vel_pub.publish(y_cmd)
            
        except rospy.ROSException:
            rospy.logwarn("获取位置失败")
            
        rate.sleep()

    rospy.sleep(1.0)

    # ========================================
    # 第五步：走到 x=4.88 到 4.93 的位置
    # ========================================
    rospy.loginfo("=== 第五步：移动到第二次上楼梯点 (x=4.88-4.93) ===")
    
    target_x_min2 = 4.89
    target_x_max2 = 4.91
    
    while not rospy.is_shutdown():
        try:
            qpos = rospy.wait_for_message('/mujoco/qpos', Float64MultiArray, timeout=2.0)
            robot_x = qpos.data[0]

            if target_x_min2 <= robot_x <= target_x_max2:
                rospy.loginfo(f"✅ 到达第二次上楼梯位置！x={robot_x:.3f}m")
                for _ in range(50):
                    cmd_vel_pub.publish(stop_cmd)
                break
            
            # 决定移动方向
            move_cmd = Twist()
            if robot_x < target_x_min2:
                # 超过目标位置（x太小），由于当前面向 180度 (负x)，需要后退
                move_cmd.linear.x = -0.04
                rospy.loginfo_throttle(1.0, f"⚠️ 超过目标位置！移动中 (后退)... x={robot_x:.3f}m -> 目标 [{target_x_min2}, {target_x_max2}]")
            else:
                # 需要减小 x，由于当前面向 180度 (负x)，需要前进
                move_cmd.linear.x = 0.08
                rospy.loginfo_throttle(1.0, f"移动中 (前进)... x={robot_x:.3f}m -> 目标 [{target_x_min2}, {target_x_max2}]")
            
            cmd_vel_pub.publish(move_cmd)
            
        except rospy.ROSException:
            rospy.logwarn("获取位置失败")
        
        rate.sleep()

    rospy.sleep(1.0)

    # ========================================
    # 第六步：再次精准对准180度
    # ========================================
    rospy.loginfo("=== 第六步：再次精准对准180度 ===")

    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"再次对准中... 当前偏航角: {math.degrees(current_yaw):.2f}度, 误差: {math.degrees(error):.2f}度")

        if abs(error) < fine_yaw_tolerance:
            rospy.loginfo(f"✅ 180度再次精确对准完成！当前偏航角: {math.degrees(current_yaw):.2f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = fine_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(0.5)

    # ========================================
    # 第七步：再次启动 stairClimbPlanner.py
    # ========================================
    rospy.loginfo("=== 第七步：再次启动上下楼梯脚本 ===")
    rospy.sleep(2.0)

    stair_process2 = subprocess.Popen(
        ['python3', stair_planner_path],
        preexec_fn=os.setsid
    )
    rospy.loginfo("✅ stairClimbPlanner.py 已第二次启动")

    rospy.loginfo("=== 等待32秒上完楼梯 ===")
    rospy.sleep(32.0)
    rospy.loginfo("✅ 第二次楼梯攀爬等待完成")

    # ========================================
    # 第八步：前进至 x=1.0 并再次对准180度
    # ========================================
    rospy.loginfo("=== 第八步：前进至 x=1.0 ===")
    
    mid_target_x = 1.0
    forward_speed = 0.05
    
    while not rospy.is_shutdown():
        try:
            qpos = rospy.wait_for_message('/mujoco/qpos', Float64MultiArray, timeout=2.0)
            robot_x = qpos.data[0]

            # 由于面向 180度 (负x方向)，前进会使 x 减小
            if robot_x <= mid_target_x:
                rospy.loginfo(f"✅ 到达中间位置！x={robot_x:.3f}m")
                for _ in range(50):
                    cmd_vel_pub.publish(stop_cmd)
                break
            
            cmd = Twist()
            cmd.linear.x = forward_speed
            cmd_vel_pub.publish(cmd)
            rospy.loginfo_throttle(1.0, f"前进中... x={robot_x:.3f}m, 目标={mid_target_x}")
            
        except rospy.ROSException:
            rospy.logwarn("获取位置失败")
            
        rate.sleep()

    rospy.sleep(1.0)

    rospy.loginfo("=== 在 x=1.0 处再次精准对准180度 ===")
    mid_yaw_tolerance = 0.01745  # 1.0度 = 1.0 * π / 180 ≈ 0.01745弧度
    mid_angular_speed = 0.08

    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"1.0m处对准中... 当前偏航角: {math.degrees(current_yaw):.2f}度, 误差: {math.degrees(error):.2f}度")

        if abs(error) < mid_yaw_tolerance:
            rospy.loginfo(f"✅ 180度精确对准完成(容差1度)！当前偏航角: {math.degrees(current_yaw):.2f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = mid_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(1.0)

    # ========================================
    # 第九步：继续前进至 x=-5.36
    # ========================================
    rospy.loginfo("=== 第九步：继续前进至 x=-5.36 ===")
    
    final_target_x = -5.36
    forward_speed = 0.05
    
    while not rospy.is_shutdown():
        try:
            qpos = rospy.wait_for_message('/mujoco/qpos', Float64MultiArray, timeout=2.0)
            robot_x = qpos.data[0]

            # 由于面向 180度 (负x方向)，前进 (linear.x > 0) 会使 x 减小
            # 我们需要走到 x = -5.36，所以当 robot_x <= -5.36 时停止
            if robot_x <= final_target_x:
                rospy.loginfo(f"✅ 到达终点位置！x={robot_x:.3f}m")
                break
            
            cmd = Twist()
            cmd.linear.x = forward_speed
            cmd_vel_pub.publish(cmd)
            rospy.loginfo_throttle(1.0, f"最后前进中... x={robot_x:.3f}m, 目标={final_target_x}")
            
        except rospy.ROSException:
            rospy.logwarn("获取位置失败")
            
        rate.sleep()

    # 先停止机器人
    rospy.loginfo("=== 到达终点，停止并停顿2秒 ===")
    for _ in range(50):
        cmd_vel_pub.publish(stop_cmd)
        rate.sleep()
        
    rospy.sleep(2.0)

    # ========================================
    # 第十步：对准-90度方向
    # ========================================
    rospy.loginfo("=== 第十步：对准-90度方向 ===")
    target_yaw_90 = -math.pi / 2

    while not rospy.is_shutdown():
        if current_yaw is None:
            continue

        error = target_yaw_90 - current_yaw
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"-90度对准中... 当前偏航角: {math.degrees(current_yaw):.2f}度, 误差: {math.degrees(error):.2f}度")

        if abs(error) < fine_yaw_tolerance:
            rospy.loginfo(f"✅ -90度精确对准完成！当前偏航角: {math.degrees(current_yaw):.2f}度")
            break

        rotate_cmd = Twist()
        rotate_cmd.angular.z = fine_angular_speed * math.copysign(1, error)
        cmd_vel_pub.publish(rotate_cmd)
        rate.sleep()

    cmd_vel_pub.publish(stop_cmd)
    rospy.sleep(1.0)

    # ========================================
    # 第十一步：持续后退
    # ========================================
    rospy.loginfo("=== 第十一步：开始持续后退 ===")
    backward_cmd = Twist()
    backward_cmd.linear.x = -0.13

    while not rospy.is_shutdown():
        cmd_vel_pub.publish(backward_cmd)
        rospy.loginfo_throttle(1.0, "持续后退中... 速度: -0.15m/s")
        rate.sleep()

    rospy.loginfo("✅ 全部巡检任务完成")
    rospy.loginfo("=== 场景一结束 ===")

    rospy.spin()


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        pass
