# simple_nav：地图定位与自主导航

该 ROS1 功能包用静态地图、AMCL 和 move_base 完成导航。`my_move_base.launch` 采用 Navfn 作为全局规划器、DWAPlannerROS 作为局部规划器，并加载 `param/` 中的全局/局部代价地图与运动参数。

## 地图

- `my_map.yaml` 是默认地图描述文件，引用 `my_map.pgm`，分辨率为 0.05 m/像素。
- `my_map1.pgm` 是附带的另一份栅格地图图像；切换地图时需要对应的 YAML 文件及正确的图像路径。

## 启动

仿真器必须先运行，且 ROS master、TF 和传感器话题可用：

```bash
# 启动地图服务、AMCL、move_base 与自动巡航节点
roslaunch simple_nav craic_sim_nav.launch

# 同时打开 RViz
roslaunch simple_nav craic_sim_nav.launch rviz:=true

# 仅导航到场景二工作台目标位姿
roslaunch simple_nav scene2_table_navigation.launch
```

`craic_sim_nav.launch` 通过点云转激光节点把 `/lidar/points` 转成 `/scan`。当前配置还使用 `/kinematic_data` 里程计话题和 `odom -> base_link` TF；AMCL 在 `map` 坐标系定位。启动后可在 RViz 使用 `2D Pose Estimate` 初始化位姿、用 `2D Nav Goal` 发送目标。RViz 默认关闭，使用 `rviz:=true` 打开，并将 Fixed Frame 设为 `map`；需要时可在 Displays 面板添加 Map、TF、LaserScan 等显示项。

`scene2_table_navigation.launch` 的目标坐标、朝向、位置容差和超时时间可通过 `target_x`、`target_y`、`target_yaw_deg`、`xy_tolerance`、`timeout` 参数修改；节点会使用 `/imu_data` 与 `/cmd_vel` 完成目标位姿调整。

## 主要文件

- `launch/my_amcl.launch`：点云转 `/scan`，启动 AMCL。
- `launch/my_move_base.launch`：代价地图和全局/局部规划配置。
- `launch/my_navigation.launch`：地图、定位、规划、可选 RViz 与巡航节点的组合入口。
- `scripts/auto_patrol.py`：多目标点巡航，发布 `/auto_patrol/completed` 完成信号。
- `scripts/scene2_table_nav.py`：场景二工作台的定位与目标姿态控制。
- `param/`：AMCL、代价地图和局部规划参数。

本包的 `package.xml` 许可证字段仍为原有 `TODO`，使用或再发布时请由项目维护者确认许可证信息。
