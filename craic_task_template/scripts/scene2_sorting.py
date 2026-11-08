#!/usr/bin/env python3
"""
场景二：分拣归档任务脚本模板

子任务一 - 料盘下架（40 分）：从料盘架夹取料盘放入物料箱
子任务二 - 分拣零件（30 分）：工作台上两种零件分类放入两侧物料箱

运行方式：
  python3 scene2_sorting.py              # 使用默认种子
  python3 scene2_sorting.py --seed 123   # 指定随机种子

可用接口：
  - /cmd_vel (geometry_msgs/Twist)               发送速度指令: linear.x=前进, linear.y=侧移, angular.z=转向
  - /kuavo_arm_traj (sensor_msgs/JointState)      手臂轨迹控制
  - /lidar/points (sensor_msgs/PointCloud2)       雷达点云数据
  - /sensors_data_raw (kuavo_msgs/sensorsData)    传感器原始数据（IMU、关节等）
  - /humanoid_controller/switch_controller        切换控制器
  - GripperController (craic_simulator)           夹爪控制
    - set_gripper_position(left, right)  左右夹爪命令 (0=张开, 255=闭合)
    - open_grippers() / close_grippers()
    - wait_for_position(target_left, target_right, tolerance, timeout)
"""

import argparse
import sys
import os
import signal
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../craic_simulator/utils'))
from sim_launcher import SimLauncher


def main():
    parser = argparse.ArgumentParser(description="场景二：分拣归档")
    parser.add_argument('--seed', type=int, default=0, help='随机种子（控制机器人初始位姿，默认0）')
    args = parser.parse_args()

    # ---- 启动仿真（自动启动 roscore + 初始化 ROS 节点） ----
    launcher = SimLauncher(scene="scene2", seed=args.seed)
    launcher.start(node_name="scene2_sorting")

    # 以下代码在 ROS 节点初始化完成后执行
    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState, PointCloud2
    from std_msgs.msg import Bool

    rospy.loginfo("=== 场景二：分拣归档任务启动 ===")

    # ---- 发布器 ----
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=10)

    rospy.sleep(1.0)  # 等待节点初始化

    # ========================================
    # 启动第二关专用导航，到零件桌前
    # ========================================
    rospy.loginfo("启动第二关专用导航...")

    nav_result = None

    def nav_callback(msg):
        nonlocal nav_result
        nav_result = bool(msg.data)
        if nav_result:
            rospy.loginfo("收到第二关导航完成信号")
        else:
            rospy.logwarn("收到第二关导航失败信号")

    rospy.Subscriber('/scene2_nav/completed', Bool, nav_callback)

    scene2_nav_script = os.path.join(
        os.path.dirname(__file__),
        "../../simple_nav/scripts/scene2_table_nav.py"
    )
    scene2_nav_script = os.path.abspath(scene2_nav_script)

    nav_process = subprocess.Popen(
        ["python3", scene2_nav_script],
        preexec_fn=os.setsid
    )

    nav_timeout = 180.0
    nav_wait_start = rospy.Time.now()

    rospy.loginfo("等待机器人导航到零件桌前...")
    while not rospy.is_shutdown():
        if nav_result is True:
            rospy.loginfo("✅ 第二关导航完成")
            break
        if nav_result is False:
            rospy.logerr("❌ 第二关导航失败，终止当前任务")
            try:
                os.killpg(nav_process.pid, signal.SIGINT)
            except Exception:
                pass
            return

        if nav_process.poll() is not None:
            rospy.logerr("❌ 第二关导航 launch 已提前退出，终止当前任务")
            return

        elapsed = (rospy.Time.now() - nav_wait_start).to_sec()
        if elapsed > nav_timeout:
            rospy.logerr("❌ 第二关导航超时，终止当前任务")
            try:
                os.killpg(nav_process.pid, signal.SIGINT)
            except Exception:
                pass
            return

        rospy.sleep(1.0)

    try:
        os.killpg(nav_process.pid, signal.SIGINT)
    except Exception:
        pass

    rospy.sleep(2.0)

    # ========================================
    # 启动第二关第一段抓取流程：零件检测 + 自动抓取
    # ========================================
    rospy.loginfo("启动第二关零件检测与第一段抓取流程...")

    child_processes = []

    def stop_child_processes():
        for proc in child_processes:
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except Exception:
                    pass

    rospy.on_shutdown(stop_child_processes)

    grab_parts_script = os.path.join(os.path.dirname(__file__), "grab_parts.py")
    grab_parts_process = subprocess.Popen(
        ["python3", grab_parts_script],
        preexec_fn=os.setsid
    )
    child_processes.append(grab_parts_process)

    rospy.loginfo("✅ 已启动零件检测与自动抓取流程")

    # ========================================
    # TODO: 在此补充第二关后续分拣流程
    # ========================================

    # 保持运行，Ctrl+C 退出
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
