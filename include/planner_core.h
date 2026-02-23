/**
 * @file planner_core.h
 * @brief 工程级无人机避障规划系统核心头文件
 * @note 已适配 new_detect_obs.h 的全局变量接口
 */

#ifndef PLANNER_CORE_H
#define PLANNER_CORE_H

#include <ros/ros.h>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <mutex>
#include <memory>
#include <cmath>
#include <algorithm>

// ROS Msgs
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <livox_ros_driver/CustomMsg.h>

// 引入你的感知模块
#include "new_detect_obs.h"

// ==========================================
// 1. 配置参数结构体 (对应 YAML)
// ==========================================
struct PlannerConfig
{
    // 系统控制
    float control_freq;
    float replan_cooldown;

    // 地图参数
    float map_size_x, map_size_y;
    float map_resolution;
    float map_origin_x, map_origin_y;
    float safety_margin; // 膨胀半径 (无人机半径 + 安全余量)

    // A* 全局规划
    float astar_heuristic_weight; // >1.0 为贪心加速
    int astar_max_iter;

    // VFH+ 局部规划
    float vfh_hist_res;      // 直方图分辨率 (度)
    float vfh_max_lookahead; // 感知距离
    float w_target;          // 目标导向权重
    float w_smooth;          // 平滑权重 (历史记忆)
    float w_obs;             // 避障权重
    float w_corridor;        // 走廊引力权重

    // 运动学限制
    float max_speed;
    float max_accel;
};

// ==========================================
// 2. 地图管理器 (双层地图：栅格 + 几何)
// ==========================================
class MapManager
{
public:
    MapManager(const PlannerConfig &cfg);

    /**
     * @brief 更新障碍物信息
     * @note 工程实现：每次更新前清空局部栅格地图，确保动态障碍物正确清除
     */
    void updateObstacles(const std::vector<Obstacle> &obs_list);

    // 几何查询：检测线段是否碰到圆形障碍物 (用于 VFH 和 后处理)
    // 精度高于栅格地图，用于局部避障判断
    bool isBlockedGeometric(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2, float robot_radius);

    // 栅格查询：用于 A*
    bool isOccupiedGrid(int gx, int gy);

    // 坐标转换
    bool worldToGrid(const Eigen::Vector2f &world, int &gx, int &gy);
    Eigen::Vector2f gridToWorld(int gx, int gy);

    // 寻找最近的空闲栅格 (用于目标点在障碍物内的情况)
    Eigen::Vector2f findNearestFreePoint(const Eigen::Vector2f &target);

    std::vector<int8_t> grid_data;       // 1D 存储的 2D 栅格地图 (0: Free, 100: Occupied)
    std::vector<Obstacle> geo_obstacles; // 几何障碍物列表

private:
    PlannerConfig cfg_;
    int width_, height_;

    // 辅助：圆形光栅化
    void rasterizeCircle(const Eigen::Vector2f &center, float r);
};

// ==========================================
// 3. A* 全局规划器 (Global Planner)
// ==========================================
class GlobalPlanner
{
public:
    GlobalPlanner(const PlannerConfig &cfg, std::shared_ptr<MapManager> map);

    // 规划核心接口
    bool plan(const Eigen::Vector2f &start, const Eigen::Vector2f &goal, std::vector<Eigen::Vector2f> &path);

    // 路径平滑 (加权平均滤波)
    std::vector<Eigen::Vector2f> smoothPath(const std::vector<Eigen::Vector2f> &raw_path);

private:
    PlannerConfig cfg_;
    std::shared_ptr<MapManager> map_;

    struct Node
    {
        int idx;
        float f, g;
        int parent;
        bool operator>(const Node &other) const { return f > other.f; }
    };
};

// ==========================================
// 4. VFH+ 局部规划器 (Local Planner)
// ==========================================
class LocalPlanner
{
public:
    LocalPlanner(const PlannerConfig &cfg, std::shared_ptr<MapManager> map);

    // 计算下一步控制指令
    // return: Vector4f (x, y, z, yaw)
    Eigen::Vector4f computeMove(const Eigen::Vector2f &curr_pos, float curr_yaw,
                                const Eigen::Vector2f &final_goal,
                                const std::vector<Eigen::Vector2f> &global_path,
                                bool &is_stuck);

private:
    PlannerConfig cfg_;
    std::shared_ptr<MapManager> map_;
    float last_selected_yaw_; // 历史记忆核心变量
    bool first_run_;

    // 寻找路径上最近的点，用于走廊引力
    Eigen::Vector2f findCorridorPoint(const Eigen::Vector2f &curr, const std::vector<Eigen::Vector2f> &path);

    // 角度归一化 (-PI ~ PI)
    float normalizeAngle(float angle);
};

// ==========================================
// 5. 顶层节点类 (ROS Interface & FSM)
// ==========================================
class UAVPlannerNode
{
public:
    UAVPlannerNode(ros::NodeHandle &nh);
    void run(); // 主循环

private:
    // ROS 通信
    ros::NodeHandle nh_;
    ros::Subscriber sub_state_, sub_odom_, sub_livox_;
    ros::Publisher pub_setpoint_;
    ros::ServiceClient client_arming_, client_mode_;

    // 内部数据
    PlannerConfig cfg_;
    mavros_msgs::State current_state_;

    // 模块实例
    std::shared_ptr<MapManager> map_manager_;
    std::shared_ptr<GlobalPlanner> global_planner_;
    std::shared_ptr<LocalPlanner> local_planner_;

    // 状态机定义
    enum class State
    {
        IDLE,
        TAKEOFF,
        PLANNING,
        AVOIDING,
        LANDING
    };
    State fsm_state_;

    // 任务变量
    Eigen::Vector2f start_pos_takeoff_; // 记录起飞点
    bool takeoff_pos_recorded_;
    std::vector<Eigen::Vector2f> global_path_;
    ros::Time last_request_time_;
    ros::Time state_start_time_;

    // 回调函数
    void cbState(const mavros_msgs::State::ConstPtr &msg);
    void cbOdom(const nav_msgs::Odometry::ConstPtr &msg);
    // Livox 回调：接收点云并转交给 new_detect_obs
    void cbLivox(const livox_ros_driver::CustomMsg::ConstPtr &msg);

    void loadParams();
    void publishSetpoint(const Eigen::Vector4f &sp);
};

#endif // PLANNER_CORE_H