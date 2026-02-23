/**
 * @file planner_core.cpp
 * @brief 工程级无人机避障规划算法实现
 */

#include "planner_core.h"

// =========================================================
// 全局变量定义 (适配 new_detect_obs.h 的 extern 声明)
// 必须在此处定义，否则链接会报错
// =========================================================
nav_msgs::Odometry local_pos;
float target_x = 0.0f;
float target_y = 0.0f;
float if_debug = 0.0f;
float init_position_x_take_off = 0.0f;
float init_position_y_take_off = 0.0f;
float init_position_z_take_off = 0.0f;
ros::NodeHandle nh; // detect_obs 中使用的全局 nh

// =========================================================
// MapManager Implementation
// =========================================================
MapManager::MapManager(const PlannerConfig &cfg) : cfg_(cfg)
{
  width_ = std::ceil(cfg.map_size_x / cfg.map_resolution);
  height_ = std::ceil(cfg.map_size_y / cfg.map_resolution);

  // 初始化栅格内存
  grid_data.resize(width_ * height_, 0);
}

void MapManager::updateObstacles(const std::vector<Obstacle> &obs_list)
{
  // 1. 更新几何层 (用于 VFH 精确计算)
  geo_obstacles = obs_list;

  // 2. 更新栅格层 (用于 A*)
  // 工程实现核心：每次更新前完全重置栅格地图为0
  // 这是处理动态障碍物最安全的方式，防止旧的障碍物残影遗留
  std::fill(grid_data.begin(), grid_data.end(), 0);

  // 3. 将所有当前帧的障碍物光栅化到地图上
  for (const auto &obs : obs_list)
  {
    // 使用配置的膨胀半径 (Obstacle半径 + 机器人半径 + 安全余量)
    rasterizeCircle(obs.position, obs.radius + cfg_.safety_margin);
  }
}

void MapManager::rasterizeCircle(const Eigen::Vector2f &center, float r)
{
  int cx, cy;
  // 如果障碍物中心在地图外，直接跳过中心点转换，但仍需考虑圆边缘可能在地图内的情况
  // 这里为了效率，如果中心太远直接忽略
  if (!worldToGrid(center, cx, cy))
  {
    // 边界检查：仅当中心距离地图边缘小于半径时才处理（此处略，假设地图足够大）
    return;
  }

  int r_cells = std::ceil(r / cfg_.map_resolution);
  int r_cells_sq = r_cells * r_cells;

  // 包围盒遍历，比全图遍历效率高
  for (int dx = -r_cells; dx <= r_cells; ++dx)
  {
    for (int dy = -r_cells; dy <= r_cells; ++dy)
    {
      int nx = cx + dx;
      int ny = cy + dy;

      // 边界检查
      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_)
      {
        // 圆方程判断
        if (dx * dx + dy * dy <= r_cells_sq)
        {
          grid_data[ny * width_ + nx] = 100; // 标记为占用
        }
      }
    }
  }
}

bool MapManager::isOccupiedGrid(int gx, int gy)
{
  if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_)
    return true; // 地图外视为障碍
  return grid_data[gy * width_ + gx] > 50;
}

bool MapManager::worldToGrid(const Eigen::Vector2f &world, int &gx, int &gy)
{
  gx = (int)((world.x() - cfg_.map_origin_x) / cfg_.map_resolution);
  gy = (int)((world.y() - cfg_.map_origin_y) / cfg_.map_resolution);
  return (gx >= 0 && gx < width_ && gy >= 0 && gy < height_);
}

Eigen::Vector2f MapManager::gridToWorld(int gx, int gy)
{
  float wx = cfg_.map_origin_x + (gx + 0.5f) * cfg_.map_resolution;
  float wy = cfg_.map_origin_y + (gy + 0.5f) * cfg_.map_resolution;
  return Eigen::Vector2f(wx, wy);
}

Eigen::Vector2f MapManager::findNearestFreePoint(const Eigen::Vector2f &target)
{
  int gx, gy;
  worldToGrid(target, gx, gy);

  // BFS 搜索最近的空闲点
  std::queue<std::pair<int, int>> q;
  q.push({gx, gy});
  std::vector<bool> visited(width_ * height_, false);
  visited[gy * width_ + gx] = true;

  int max_steps = 100; // 限制搜索范围，防止无限循环
  int steps = 0;

  while (!q.empty() && steps < max_steps)
  {
    auto curr = q.front();
    q.pop();
    int cx = curr.first;
    int cy = curr.second;

    if (!isOccupiedGrid(cx, cy))
    {
      return gridToWorld(cx, cy);
    }

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};
    for (int i = 0; i < 4; ++i)
    {
      int nx = cx + dx[i];
      int ny = cy + dy[i];
      if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_)
      {
        int idx = ny * width_ + nx;
        if (!visited[idx])
        {
          visited[idx] = true;
          q.push({nx, ny});
        }
      }
    }
    steps++;
  }
  return target; // 如果找不到，无奈返回原点
}

// =========================================================
// GlobalPlanner Implementation (Heuristic A*)
// =========================================================
GlobalPlanner::GlobalPlanner(const PlannerConfig &cfg, std::shared_ptr<MapManager> map)
    : cfg_(cfg), map_(map) {}

bool GlobalPlanner::plan(const Eigen::Vector2f &start, const Eigen::Vector2f &goal, std::vector<Eigen::Vector2f> &path)
{
  path.clear();
  int start_gx, start_gy;

  // 1. 起点校验
  if (!map_->worldToGrid(start, start_gx, start_gy))
  {
    ROS_ERROR("[Global] Start position out of map bounds!");
    return false;
  }

  // 2. 终点校验与修正
  // 工程细节：如果目标点刚好在墙里，不要直接报错，而是去目标旁边最近的空地
  Eigen::Vector2f safe_goal = goal;
  int goal_gx, goal_gy;
  map_->worldToGrid(goal, goal_gx, goal_gy); // 计算原始栅格

  if (map_->isOccupiedGrid(goal_gx, goal_gy))
  {
    ROS_WARN("[Global] Goal is inside obstacle, searching for nearest free point...");
    safe_goal = map_->findNearestFreePoint(goal);
    map_->worldToGrid(safe_goal, goal_gx, goal_gy);
  }

  // A* 数据结构
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
  std::vector<int> g_score(map_->grid_data.size(), 1e9);
  std::vector<int> parent(map_->grid_data.size(), -1);
  std::vector<bool> closed(map_->grid_data.size(), false);

  int width = (int)(cfg_.map_size_x / cfg_.map_resolution);
  int start_idx = start_gy * width + start_gx;
  int goal_idx = goal_gy * width + goal_gx;

  g_score[start_idx] = 0;
  open_set.push({start_idx, 0.0f, 0.0f, -1});

  int iter = 0;
  bool found = false;
  const int dx[] = {1, -1, 0, 0, 1, 1, -1, -1}; // 8 连通
  const int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};
  const float move_cost[] = {1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414};

  while (!open_set.empty() && iter++ < cfg_.astar_max_iter)
  {
    Node current = open_set.top();
    open_set.pop();

    if (closed[current.idx])
      continue;
    closed[current.idx] = true;

    if (current.idx == goal_idx)
    {
      found = true;
      break;
    }

    int cx = current.idx % width;
    int cy = current.idx / width;

    for (int i = 0; i < 8; ++i)
    {
      int nx = cx + dx[i];
      int ny = cy + dy[i];

      if (map_->isOccupiedGrid(nx, ny))
        continue;

      int n_idx = ny * width + nx;
      float new_g = g_score[current.idx] + move_cost[i];

      if (new_g < g_score[n_idx])
      {
        g_score[n_idx] = new_g;
        parent[n_idx] = current.idx;

        // 启发式：加权的欧几里得距离，权重越高搜索越快但非最优
        float dist = std::sqrt(std::pow(nx - goal_gx, 2) + std::pow(ny - goal_gy, 2));
        float h = dist * cfg_.astar_heuristic_weight;

        open_set.push({n_idx, new_g + h, new_g, current.idx});
      }
    }
  }

  if (found)
  {
    int curr = goal_idx;
    while (curr != -1)
    {
      int cx = curr % width;
      int cy = curr / width;
      path.push_back(map_->gridToWorld(cx, cy));
      curr = parent[curr];
    }
    std::reverse(path.begin(), path.end());
    return true;
  }
  else
  {
    ROS_WARN("[Global] A* failed to find path (Iter: %d)", iter);
    return false;
  }
}

std::vector<Eigen::Vector2f> GlobalPlanner::smoothPath(const std::vector<Eigen::Vector2f> &raw_path)
{
  if (raw_path.size() < 3)
    return raw_path;

  std::vector<Eigen::Vector2f> smoothed;
  smoothed.push_back(raw_path.front());

  // 简单的权重平滑处理
  for (size_t i = 1; i < raw_path.size() - 1; ++i)
  {
    Eigen::Vector2f p_prev = raw_path[i - 1];
    Eigen::Vector2f p_curr = raw_path[i];
    Eigen::Vector2f p_next = raw_path[i + 1];

    // P' = 0.25*Prev + 0.5*Curr + 0.25*Next
    Eigen::Vector2f p_new = 0.25 * p_prev + 0.5 * p_curr + 0.25 * p_next;
    smoothed.push_back(p_new);
  }
  smoothed.push_back(raw_path.back());
  return smoothed;
}

// =========================================================
// LocalPlanner Implementation (VFH+)
// =========================================================
LocalPlanner::LocalPlanner(const PlannerConfig &cfg, std::shared_ptr<MapManager> map)
    : cfg_(cfg), map_(map), first_run_(true) {}

float LocalPlanner::normalizeAngle(float angle)
{
  while (angle > M_PI)
    angle -= 2.0 * M_PI;
  while (angle < -M_PI)
    angle += 2.0 * M_PI;
  return angle;
}

Eigen::Vector2f LocalPlanner::findCorridorPoint(const Eigen::Vector2f &curr, const std::vector<Eigen::Vector2f> &path)
{
  float min_d = 1e9;
  Eigen::Vector2f closest = curr;

  // 线性搜索最近点 (由于路径点数通常<200，效率可接受)
  for (const auto &p : path)
  {
    float d = (p - curr).norm();
    if (d < min_d)
    {
      min_d = d;
      closest = p;
    }
  }
  return closest;
}

Eigen::Vector4f LocalPlanner::computeMove(const Eigen::Vector2f &curr_pos, float curr_yaw,
                                          const Eigen::Vector2f &final_goal,
                                          const std::vector<Eigen::Vector2f> &global_path,
                                          bool &is_stuck)
{
  is_stuck = false;
  if (first_run_)
  {
    last_selected_yaw_ = curr_yaw;
    first_run_ = false;
  }

  // 1. 构建极坐标直方图 (Polar Histogram)
  int hist_size = 360 / (int)cfg_.vfh_hist_res;
  std::vector<float> histogram(hist_size, 0.0f);

  // 使用几何地图进行精确计算 (避免栅格化导致的精度损失)
  for (const auto &obs : map_->geo_obstacles)
  {
    Eigen::Vector2f vec = obs.position - curr_pos; // 注意 Obstacle 结构体名为 position
    float dist = vec.norm();

    // 超出感知范围忽略
    if (dist > cfg_.vfh_max_lookahead)
      continue;

    // 计算障碍物张角
    float angle_center = std::atan2(vec.y(), vec.x());
    float obs_r = obs.radius + cfg_.safety_margin; // 包含膨胀

    // 反正弦计算张角，注意输入范围
    float val = std::min(1.0f, obs_r / dist);
    float angle_width = std::asin(val);

    int bin_center = (int)((normalizeAngle(angle_center) + M_PI) / (2 * M_PI) * hist_size);
    int bin_width = (int)(angle_width / (2 * M_PI) * hist_size) + 1; // 至少占一个bin

    // 填充直方图 (高代价)
    for (int i = bin_center - bin_width; i <= bin_center + bin_width; ++i)
    {
      int idx = (i + hist_size) % hist_size;
      // 距离越近，代价越高，且叠加
      histogram[idx] += 100.0f * (1.0f - dist / cfg_.vfh_max_lookahead);
    }
  }

  // 2. 选择最佳扇区 (Cost Function)
  Eigen::Vector2f goal_vec = final_goal - curr_pos;
  float target_angle = std::atan2(goal_vec.y(), goal_vec.x());

  // 走廊引力计算
  Eigen::Vector2f corridor_pt = findCorridorPoint(curr_pos, global_path);
  Eigen::Vector2f corr_vec = corridor_pt - curr_pos;
  float corridor_angle = std::atan2(corr_vec.y(), corr_vec.x());
  float dist_error = corr_vec.norm();

  float best_cost = 1e9;
  int best_bin = -1;

  for (int i = 0; i < hist_size; ++i)
  {
    // 如果该方向障碍物代价过高，直接忽略该方向（二值化思想）
    if (histogram[i] > 80.0f)
      continue;

    float bin_angle = -M_PI + i * (2 * M_PI / hist_size) + (M_PI / hist_size * 0.5f); // bin中心

    // 代价1：目标偏差
    float cost_target = std::abs(normalizeAngle(bin_angle - target_angle));

    // 代价2：历史平滑 (hysteresis)
    float cost_smooth = std::abs(normalizeAngle(bin_angle - last_selected_yaw_));

    // 代价3：走廊引力
    float cost_corridor = std::abs(normalizeAngle(bin_angle - corridor_angle));

    // 走廊权重的动态调整：离走廊越远，拉回的力越大
    float current_corridor_w = cfg_.w_corridor * std::min(2.0f, dist_error * 2.0f);

    float total_cost = cfg_.w_target * cost_target +
                       cfg_.w_smooth * cost_smooth +
                       current_corridor_w * cost_corridor +
                       cfg_.w_obs * (histogram[i] / 100.0f); // 归一化障碍物代价

    if (total_cost < best_cost)
    {
      best_cost = total_cost;
      best_bin = i;
    }
  }

  // 3. 死锁检测
  if (best_bin == -1)
  {
    is_stuck = true;
    return Eigen::Vector4f(curr_pos.x(), curr_pos.y(), 0, curr_yaw);
  }

  // 4. 生成控制指令
  float best_yaw = -M_PI + best_bin * (2 * M_PI / hist_size) + (M_PI / hist_size * 0.5f);
  last_selected_yaw_ = best_yaw;

  // 速度平滑：如果需要大幅度转弯，减速
  float speed = std::min(cfg_.max_speed, goal_vec.norm());
  if (std::abs(normalizeAngle(best_yaw - curr_yaw)) > 0.5)
  {
    speed *= 0.5f;
  }

  // 前向积分预测
  float dt = 0.5f;
  float next_x = curr_pos.x() + std::cos(best_yaw) * speed * dt;
  float next_y = curr_pos.y() + std::sin(best_yaw) * speed * dt;

  return Eigen::Vector4f(next_x, next_y, 0, best_yaw);
}

// =========================================================
// UAVPlannerNode Implementation (State Machine)
// =========================================================
UAVPlannerNode::UAVPlannerNode(ros::NodeHandle &nh) : nh_(nh), fsm_state_(State::IDLE)
{
  loadParams();

  // 初始化模块
  map_manager_ = std::make_shared<MapManager>(cfg_);
  global_planner_ = std::make_shared<GlobalPlanner>(cfg_, map_manager_);
  local_planner_ = std::make_shared<LocalPlanner>(cfg_, map_manager_);

  // 订阅与发布
  sub_state_ = nh_.subscribe<mavros_msgs::State>("mavros/state", 10, &UAVPlannerNode::cbState, this);
  sub_odom_ = nh_.subscribe<nav_msgs::Odometry>("/mavros/local_position/odom", 10, &UAVPlannerNode::cbOdom, this);
  sub_livox_ = nh_.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", 10, &UAVPlannerNode::cbLivox, this);

  pub_setpoint_ = nh_.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 10);
  client_arming_ = nh_.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
  client_mode_ = nh_.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");

  takeoff_pos_recorded_ = false;
  last_request_time_ = ros::Time::now();

  ROS_INFO("UAV Planner Node Initialized.");
}

void UAVPlannerNode::loadParams()
{
  // ROS Params Loading (从 YAML 读取)
  nh_.param<float>("planner/control_freq", cfg_.control_freq, 20.0);
  nh_.param<float>("planner/replan_cooldown", cfg_.replan_cooldown, 2.0);
  nh_.param<float>("planner/map_size_x", cfg_.map_size_x, 20.0);
  nh_.param<float>("planner/map_size_y", cfg_.map_size_y, 20.0);
  nh_.param<float>("planner/map_resolution", cfg_.map_resolution, 0.1);
  nh_.param<float>("planner/map_origin_x", cfg_.map_origin_x, -10.0);
  nh_.param<float>("planner/map_origin_y", cfg_.map_origin_y, -10.0);
  nh_.param<float>("planner/safety_margin", cfg_.safety_margin, 0.4);

  nh_.param<float>("planner/astar/heuristic_weight", cfg_.astar_heuristic_weight, 1.5);
  nh_.param<int>("planner/astar/max_iter", cfg_.astar_max_iter, 5000);

  nh_.param<float>("planner/vfh/max_lookahead", cfg_.vfh_max_lookahead, 3.5);
  nh_.param<float>("planner/vfh/hist_res", cfg_.vfh_hist_res, 5.0);
  nh_.param<float>("planner/vfh/w_target", cfg_.w_target, 2.0);
  nh_.param<float>("planner/vfh/w_smooth", cfg_.w_smooth, 3.0);
  nh_.param<float>("planner/vfh/w_obs", cfg_.w_obs, 5.0);
  nh_.param<float>("planner/vfh/w_corridor", cfg_.w_corridor, 2.0);

  nh_.param<float>("planner/max_speed", cfg_.max_speed, 0.8);

  // 从参数服务器加载目标点
  nh_.param<float>("target_x", target_x, 5.0f);
  nh_.param<float>("target_y", target_y, 0.0f);
  nh_.param<float>("if_debug", if_debug, 0.0f);
}

void UAVPlannerNode::cbState(const mavros_msgs::State::ConstPtr &msg) { current_state_ = *msg; }

void UAVPlannerNode::cbOdom(const nav_msgs::Odometry::ConstPtr &msg)
{
  // 1. 更新内部 Odom
  // 2. 更新全局变量 local_pos，供 detect_obs 使用
  local_pos = *msg;
}

void UAVPlannerNode::cbLivox(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
  // 1. 调用 detect_obs 的全局包装函数处理点云
  livox_cb_wrapper(msg);

  // 2. 从 detect_obs 获取处理后的障碍物列表 (accessing external global variable)
  // 工程实现：在规划主循环前更新地图，这里仅触发感知处理
  // 真正的地图更新在 run() 中进行，确保线程安全或者逻辑顺序
}

void UAVPlannerNode::run()
{
  ros::Rate rate(cfg_.control_freq);

  while (ros::ok())
  {
    ros::spinOnce();

    // 0. 同步数据：从 detect_obs 的全局变量获取障碍物并更新地图
    // 注意：obstacles 是 new_detect_obs.h 中定义的全局变量
    map_manager_->updateObstacles(obstacles);

    // 获取当前无人机状态
    Eigen::Vector2f curr_pos(local_pos.pose.pose.position.x, local_pos.pose.pose.position.y);
    float curr_z = local_pos.pose.pose.position.z;
    tf::Quaternion q;
    tf::quaternionMsgToTF(local_pos.pose.pose.orientation, q);
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

    // 如果未记录起飞点，记录之 (前提是已经有高度)
    if (!takeoff_pos_recorded_ && curr_z > 0.1)
    {
      init_position_x_take_off = curr_pos.x();
      init_position_y_take_off = curr_pos.y();
      init_position_z_take_off = curr_z;
      start_pos_takeoff_ = curr_pos;
      takeoff_pos_recorded_ = true;
      ROS_INFO("Takeoff position recorded: (%.2f, %.2f)", curr_pos.x(), curr_pos.y());
    }

    // 目标高度设定 (相对起飞高度)
    float target_z_abs = init_position_z_take_off + 0.8f;

    // 状态机逻辑
    switch (fsm_state_)
    {
    case State::IDLE:
    {
      // 发送当前位置作为 setpoint 防止漂移
      Eigen::Vector4f hold(curr_pos.x(), curr_pos.y(), curr_z, yaw);
      publishSetpoint(hold);

      // 自动切 OFFBOARD 和 解锁
      if (current_state_.mode != "OFFBOARD" && (ros::Time::now() - last_request_time_ > ros::Duration(5.0)))
      {
        mavros_msgs::SetMode offb_set_mode;
        offb_set_mode.request.custom_mode = "OFFBOARD";
        if (client_mode_.call(offb_set_mode) && offb_set_mode.response.mode_sent)
        {
          ROS_INFO("Offboard enabled");
        }
        last_request_time_ = ros::Time::now();
      }
      else if (current_state_.mode == "OFFBOARD" && !current_state_.armed && (ros::Time::now() - last_request_time_ > ros::Duration(5.0)))
      {
        mavros_msgs::CommandBool arm_cmd;
        arm_cmd.request.value = true;
        if (client_arming_.call(arm_cmd) && arm_cmd.response.success)
        {
          ROS_INFO("Vehicle armed");
        }
        last_request_time_ = ros::Time::now();
      }

      if (current_state_.mode == "OFFBOARD" && current_state_.armed && takeoff_pos_recorded_)
      {
        fsm_state_ = State::TAKEOFF;
        ROS_INFO("Armed & Offboard -> Switching to TAKEOFF");
      }
      break;
    }

    case State::TAKEOFF:
    {
      // 原地起飞到目标高度
      Eigen::Vector4f sp(start_pos_takeoff_.x(), start_pos_takeoff_.y(), target_z_abs, yaw);
      publishSetpoint(sp);

      if (std::abs(curr_z - target_z_abs) < 0.15)
      {
        fsm_state_ = State::PLANNING;
        state_start_time_ = ros::Time::now();
        ROS_INFO("Takeoff Complete -> Switching to PLANNING");
      }
      break;
    }

    case State::PLANNING:
    {
      // 计算世界坐标系下的目标点
      Eigen::Vector2f goal_world(
          init_position_x_take_off + target_x,
          init_position_y_take_off + target_y);

      std::vector<Eigen::Vector2f> raw_path;
      if (global_planner_->plan(curr_pos, goal_world, raw_path))
      {
        global_path_ = global_planner_->smoothPath(raw_path);
        fsm_state_ = State::AVOIDING;
        ROS_INFO("A* Plan Success (%lu pts) -> Switching to AVOIDING", global_path_.size());
      }
      else
      {
        ROS_WARN_THROTTLE(1.0, "A* Plan Failed! Obstacle blocking? Retrying...");
        // 悬停等待
        publishSetpoint(Eigen::Vector4f(curr_pos.x(), curr_pos.y(), target_z_abs, yaw));
      }
      break;
    }

    case State::AVOIDING:
    {
      Eigen::Vector2f goal_world(
          init_position_x_take_off + target_x,
          init_position_y_take_off + target_y);
      bool is_stuck = false;

      // VFH+ 局部避障
      Eigen::Vector4f next_move = local_planner_->computeMove(curr_pos, yaw, goal_world, global_path_, is_stuck);
      next_move[2] = target_z_abs; // 保持高度

      if (is_stuck)
      {
        ROS_ERROR("VFH Stuck! Triggering Replan...");
        fsm_state_ = State::PLANNING;
      }
      else
      {
        publishSetpoint(next_move);
      }

      // 超时重规划 (防止局部极小值)
      if ((ros::Time::now() - state_start_time_).toSec() > 30.0)
      {
        ROS_WARN("Avoiding Timeout -> Replan");
        fsm_state_ = State::PLANNING;
        state_start_time_ = ros::Time::now();
      }

      // 到达判定
      if ((curr_pos - goal_world).norm() < 0.4)
      {
        ROS_INFO("Target Reached -> Switching to LANDING");
        fsm_state_ = State::LANDING;
      }
      break;
    }

    case State::LANDING:
    {
      // 缓降
      Eigen::Vector4f land_sp(curr_pos.x(), curr_pos.y(), curr_z - 0.15f, yaw);
      publishSetpoint(land_sp);

      if (curr_z < init_position_z_take_off + 0.1)
      {
        ROS_INFO("Landed.");
        // 可选：发送 Disarm 指令
        // fsm_state_ = State::IDLE;
      }
      break;
    }
    }

    rate.sleep();
  }
}

void UAVPlannerNode::publishSetpoint(const Eigen::Vector4f &sp)
{
  mavros_msgs::PositionTarget msg;
  msg.header.stamp = ros::Time::now();
  msg.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
  // 忽略速度和加速度控制，仅控制位置+Yaw
  msg.type_mask = mavros_msgs::PositionTarget::IGNORE_VX |
                  mavros_msgs::PositionTarget::IGNORE_VY |
                  mavros_msgs::PositionTarget::IGNORE_VZ |
                  mavros_msgs::PositionTarget::IGNORE_AFX |
                  mavros_msgs::PositionTarget::IGNORE_AFY |
                  mavros_msgs::PositionTarget::IGNORE_AFZ |
                  mavros_msgs::PositionTarget::IGNORE_YAW_RATE;

  msg.position.x = sp[0];
  msg.position.y = sp[1];
  msg.position.z = sp[2];
  msg.yaw = sp[3];
  pub_setpoint_.publish(msg);
}

// =========================================================
// Main Function
// =========================================================
int main(int argc, char **argv)
{
  ros::init(argc, argv, "astar_node"); // 节点名保持原样或改为 uav_planner
  ros::NodeHandle nh("~");

  // 实例化节点类
  UAVPlannerNode node(nh);
  node.run();

  return 0;
}