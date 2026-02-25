#ifndef NEW_DETECT_OBS_H
#define NEW_DETECT_OBS_H
#include <ros/ros.h>
#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <nav_msgs/Odometry.h>
#include <assert.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Header.h>
#include <Eigen/StdVector>
#include <tf/transform_listener.h>

extern float target_x;
extern float target_y;
//extern ros::NodeHandle nh;
extern nav_msgs::Odometry local_pos;
extern float if_debug;
extern float init_position_x_take_off;
extern float init_position_y_take_off;
extern float init_position_z_take_off;

struct Obstacle
{
    int id;
    Eigen::Vector2f position;
    float radius;
    int missed_count;
};
std::vector<Obstacle> obstacles;
const float matching_distance_threshold{1.0f};
const int max_missed_frames{5};

constexpr float height_threshold_value{0.3};
const float voxel_size_{0.04f}; // 体素滤波尺寸
const float radius{0.5f};
const int min_neighbors{3};
int max_cluster_size{150};        // 最大簇大小，
float min_obstacle_radius_{0.1f}; // 最小障碍物半径
float max_obstacle_radius_{3.f};  // 最大障碍物半径
float drone_radius{0.f};

class detect_obs
{

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_world;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel;
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxel_grid;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_dense;
    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    std::vector<int> indices;
    std::vector<float> distances;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr cluster_tree;
    std::vector<pcl::PointIndices> cluster_indices;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    ros::Publisher obstacle_marker_pub;
    static std::vector<Obstacle> historical_obstacles;
    static int next_id;

public:
    static detect_obs &getInstance()
    {
        static detect_obs instance;
        return instance;
    }

    void livox_custom_cb(const livox_ros_driver::CustomMsg::ConstPtr &msg)
    {

        static bool publisher_initialized = false;
        if (!publisher_initialized)
        {
            ros::NodeHandle nh;
            obstacle_marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/obstacle_markers", 1);
            publisher_initialized = true;
        }

        if (!cloud_world)
            cloud_world.reset(new pcl::PointCloud<pcl::PointXYZ>);
        if (!cloud_voxel)
            cloud_voxel.reset(new pcl::PointCloud<pcl::PointXYZ>);

        convertLivoxToPCL(msg, cloud_world);

        // 3. 体素滤波并投影到XY平面
        voxelFilterAndProject(cloud_world, cloud_voxel);

        // // // 4. 筛选密集体素(障碍物候选)
        filterDenseVoxels(cloud_voxel, cloud_dense);

        euclideanClustering(cloud_dense, clusters);

        publishObstacleMarkers(clusters);

        Eigen::Vector2f drone_pos(local_pos.pose.pose.position.x,
                                  local_pos.pose.pose.position.y);
        Eigen::Vector2f target_pos(target_x + init_position_x_take_off, target_y + init_position_y_take_off); // 修复：目标点转绝对坐标

        processObstaclesWithMemory(clusters, drone_pos, target_pos);

        logTheInfo(msg);
    }

private:
    detect_obs()
    {
        cloud_world.reset(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_voxel.reset(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_dense.reset(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_cluster.reset(new pcl::PointCloud<pcl::PointXYZ>);
        cluster_tree.reset(new pcl::search::KdTree<pcl::PointXYZ>);
        voxel_grid.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    }
    detect_obs(const detect_obs &) = delete;
    detect_obs &operator=(const detect_obs &) = delete;

    static void convertLivoxToPCL(const livox_ros_driver::CustomMsg::ConstPtr &livox_msg,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_cloud)
    {
        pcl_cloud->clear();
        const size_t point_num = livox_msg->point_num;
        pcl_cloud->is_dense = true;

        float plx = local_pos.pose.pose.position.x;
        float ply = local_pos.pose.pose.position.y;
        float pyaw = static_cast<float>(tf::getYaw(local_pos.pose.pose.orientation));
        // ROS_INFO("pyaw:%.2f",pyaw);
        const livox_ros_driver::CustomPoint *livox_points = livox_msg->points.data();
        pcl::PointXYZ *pcl_points = pcl_cloud->points.data();

        const float cy = cosf(pyaw);
        const float sy = sinf(pyaw);

        for (size_t i = 0; i < point_num; ++i)
        {
            if (livox_points[i].z >= -height_threshold_value && livox_points[i].z <= height_threshold_value)
            {
                float x = livox_points[i].x;
                float y = livox_points[i].y;
                float px = x * cy - y * sy + plx;
                float py = x * sy + y * cy + ply;
                pcl_cloud->push_back({px, py, 0});
            }
        }
    }

    // 体素滤波并投影到XY平面
    void voxelFilterAndProject(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr &output)
    {
        output->clear();
        voxel_grid.setInputCloud(input);
        voxel_grid.filter(*output);
    }

    void filterDenseVoxels(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, pcl::PointCloud<pcl::PointXYZ>::Ptr &output)
    {
        if (!input || input->empty())
        {
            output->clear();
            return;
        }
        kdtree.setInputCloud(input);

        // 检查点云是否有效
        for (const auto &point : input->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            {
                std::cout << "警告：点云包含NaN或Inf值！" << std::endl;
                output->clear();
                return;
            }
        }
        output->clear();
        output->reserve(input->size());
        for (size_t i = 0; i < input->size(); ++i)
        {
            indices.clear();
            distances.clear();
            int k = kdtree.radiusSearch(input->points[i], radius, indices, distances);
            if (k >= min_neighbors)
            {
                output->push_back(input->points[i]);
            }
        }
    }

    void euclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input,
                             std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters)
    {
        clusters.clear();
        cluster_indices.clear();
        if (input->empty())
            return;

        // 设置聚类参数
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.3f); // 聚类距离阈值（米）
        ec.setMinClusterSize(5);      // 最小点数（防噪点）
        ec.setMaxClusterSize(10000);  // 最大点数（防墙体）
        ec.setSearchMethod(cluster_tree);
        ec.setInputCloud(input);

        std::vector<pcl::PointIndices> local_cluster_indices;
        ec.extract(local_cluster_indices);

        // 转换索引为点云
        for (const auto &indices : local_cluster_indices)
        {
            if (indices.indices.size() > max_cluster_size)
            {
                continue;
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            cluster->reserve(indices.indices.size());
            for (int idx : indices.indices)
            {

                cluster->push_back(input->points[idx]);
            }
            clusters.push_back(cluster);
        }
    }

    void publishObstacleMarkers(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters)
    {
        visualization_msgs::MarkerArray marker_array;

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            if (clusters[i]->empty())
                continue;

            // 1. 计算簇的中心 (x, y)
            float sum_x = 0, sum_y = 0;
            for (const auto &pt : clusters[i]->points)
            {
                sum_x += pt.x;
                sum_y += pt.y;
            }
            float cx = sum_x / clusters[i]->size();
            float cy = sum_y / clusters[i]->size();

            // 2. 创建圆圈 marker（表示障碍物范围）
            visualization_msgs::Marker circle_marker;
            circle_marker.header.frame_id = "map"; // 或你的世界坐标系
            circle_marker.header.stamp = ros::Time::now();
            circle_marker.ns = "obstacle_circles";
            circle_marker.id = i;
            circle_marker.type = visualization_msgs::Marker::CYLINDER;
            circle_marker.action = visualization_msgs::Marker::ADD;
            circle_marker.pose.position.x = cx;
            circle_marker.pose.position.y = cy;
            circle_marker.pose.position.z = 0.2; // 稍微抬高避免被地面遮挡
            circle_marker.pose.orientation.w = 1.0;
            circle_marker.scale.x = 0.2; // 直径
            circle_marker.scale.y = 0.2;
            circle_marker.scale.z = 0.4; // 高度
            circle_marker.color.r = 1.0;
            circle_marker.color.a = 0.5; // 半透明
            marker_array.markers.push_back(circle_marker);

            // 3. 创建文本 marker（显示坐标）
            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = "map";
            text_marker.header.stamp = ros::Time::now();
            text_marker.ns = "obstacle_labels";
            text_marker.id = i + 1000; // 避免 ID 冲突
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            text_marker.pose.position.x = cx;
            text_marker.pose.position.y = cy;
            text_marker.pose.position.z = 0.5; // 更高一点
            text_marker.pose.orientation.w = 1.0;
            text_marker.scale.z = 0.2; // 字体大小
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            text_marker.text = "ID:" + std::to_string(i) + "\n(" +
                               std::to_string(cx) + ", " +
                               std::to_string(cy) + ")";
            marker_array.markers.push_back(text_marker);
        }

        // 清除上一帧的 markers（可选）
        visualization_msgs::Marker clear_marker;
        clear_marker.header.frame_id = "map";
        clear_marker.header.stamp = ros::Time::now();
        clear_marker.ns = "obstacle_circles";
        clear_marker.id = 0;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        // 如果你用 DELETEALL，上面就不需要逐个管理 ID，但会闪一下
        // 这里我们不用 DELETEALL，靠 ID 更新
        marker_array.markers.push_back(clear_marker);

        visualization_msgs::Marker clear_text_marker;
        clear_text_marker.header.frame_id = "map";
        clear_text_marker.header.stamp = ros::Time::now();
        clear_text_marker.ns = "obstacle_labels";
        clear_text_marker.id = 0;
        clear_text_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear_text_marker);

        obstacle_marker_pub.publish(marker_array);
    }

    // 修复10：正确实现ccw函数，修复线段相交判断
    bool ccw(const Eigen::Vector2f &a, const Eigen::Vector2f &b, const Eigen::Vector2f &c)
    {
        return (b.x() - a.x()) * (c.y() - a.y()) > (b.y() - a.y()) * (c.x() - a.x());
    }

    bool segmentsIntersect(
        const Eigen::Vector2f &p1, const Eigen::Vector2f &p2,
        const Eigen::Vector2f &q1, const Eigen::Vector2f &q2)
    {
        if (!std::isfinite(p1.x()) || !std::isfinite(p1.y()) ||
            !std::isfinite(p2.x()) || !std::isfinite(p2.y()) ||
            !std::isfinite(q1.x()) || !std::isfinite(q1.y()) ||
            !std::isfinite(q2.x()) || !std::isfinite(q2.y()))
        {
            return false;
        }
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) && (ccw(p1, p2, q1) != ccw(p1, p2, q2));
    }

    // 判断点簇是否与路径 (drone → target) 相交
    bool clusterIntersectsPath(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
        const Eigen::Vector2f &drone_pos,
        const Eigen::Vector2f &target_pos)
    {

        // 获取簇的包围盒
        float min_x = FLT_MAX, max_x = -FLT_MAX;
        float min_y = FLT_MAX, max_y = -FLT_MAX;
        for (const auto &pt : cluster->points)
        {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }

        // 用包围盒的四条边与路径线段相交检测
        Eigen::Vector2f box_corners[4] = {
            {min_x, min_y}, {max_x, min_y}, {max_x, max_y}, {min_x, max_y}};

        // 检查路径是否穿过包围盒
        for (int i = 0; i < 4; i++)
        {
            if (segmentsIntersect(drone_pos, target_pos,
                                  box_corners[i], box_corners[(i + 1) % 4]))
            {
                return true;
            }
        }

        // 额外检查：路径起点/终点在包围盒内
        if (drone_pos.x() >= min_x && drone_pos.x() <= max_x &&
            drone_pos.y() >= min_y && drone_pos.y() <= max_y)
            return true;
        if (target_pos.x() >= min_x && target_pos.x() <= max_x &&
            target_pos.y() >= min_y && target_pos.y() <= max_y)
            return true;

        return false;
    }

    // 拟合最小包围圆（适用于任意形状）
    void fitMinimumEnclosingCircle(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
        Eigen::Vector2f &center,
        float &radius)
    {

        // 简化版：用质心 + 最远点距离
        center = Eigen::Vector2f(0, 0);
        for (const auto &pt : cluster->points)
        {
            center.x() += pt.x;
            center.y() += pt.y;
        }
        center /= static_cast<float>(cluster->size());

        radius = 0;
        for (const auto &pt : cluster->points)
        {
            float dx = pt.x - center.x();
            float dy = pt.y - center.y();
            radius = std::max(radius, std::sqrt(dx * dx + dy * dy) + drone_radius);
        }
    }

    // 主障碍物验证函数
    bool validateAndFitObstacle(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
        const Eigen::Vector2f &drone_pos,
        const Eigen::Vector2f &target_pos,
        Eigen::Vector2f &center,
        float &radius)
    {

        if (cluster->empty())
            return false;

        if (cluster->size() <= 10)
        {
            fitMinimumEnclosingCircle(cluster, center, radius);
            return true;
        }

        // 1. 计算圆形度
        Eigen::Vector2f temp_center;
        float temp_radius, circularity;
        computeCircularity(cluster, temp_center, temp_radius, circularity);

        // 2. 弧形障碍物：直接接受
        if (circularity < 0.6f)
        {
            fitMinimumEnclosingCircle(cluster, center, radius);
            return true;
        }

        // 3. 直线型障碍物：检查是否阻挡路径
        if (clusterIntersectsPath(cluster, drone_pos, target_pos))
        {
            fitMinimumEnclosingCircle(cluster, center, radius);
            return true;
        }

        // 4. 其他情况：忽略

        // fitMinimumEnclosingCircle(cluster, center, radius);
        // return true;

        return false;
    }

    // 辅助函数：计算圆形度
    void computeCircularity(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
        Eigen::Vector2f &center,
        float &radius,
        float &circularity)
    {

        center = Eigen::Vector2f(0, 0);
        for (const auto &pt : cluster->points)
        {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y))
                continue;
            center.x() += pt.x;
            center.y() += pt.y;
        }
        if (cluster->size() == 0)
        {
            circularity = 10.0f;
            radius = 0;
            return;
        }
        center /= static_cast<float>(cluster->size());

        std::vector<float> radii;
        for (const auto &pt : cluster->points)
        {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y))
                continue;
            float dx = pt.x - center.x();
            float dy = pt.y - center.y();
            radii.push_back(std::sqrt(dx * dx + dy * dy));
        }

        if (radii.empty())
        {
            circularity = 10.0f;
            radius = 0;
            return;
        }

        radius = 0;
        for (float r : radii)
            radius += r;
        radius /= radii.size();

        float var = 0;
        for (float r : radii)
        {
            float diff = r - radius;
            var += diff * diff;
        }
        float std_dev = std::sqrt(var / radii.size());
        circularity = std_dev / (radius + 1e-6f);
    }

    void processObstaclesWithMemory(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &current_clusters,
        const Eigen::Vector2f &drone_position,
        const Eigen::Vector2f &target_position)
    {

        // 清空当前帧的障碍物列表
        obstacles.clear();
        // 创建标记数组，记录哪些历史障碍物在当前帧被匹配到
        std::vector<bool> historical_obstacle_matched(historical_obstacles.size(), false);

        // 1. 处理新检测到的障碍物，并尝试与历史障碍物匹配
        std::vector<Obstacle> new_detections;
        for (auto &cluster : current_clusters)
        {
            Eigen::Vector2f obstacle_center;
            float obstacle_radius;

            if (validateAndFitObstacle(cluster, drone_position, target_position, obstacle_center, obstacle_radius))
            {
                // 半径过滤（避免过小/过大）
                if (obstacle_radius >= min_obstacle_radius_ && obstacle_radius <= max_obstacle_radius_ && obstacle_center[0] <= drone_position[0] + 6.0f && obstacle_center[0] >= drone_position[0] - 1.5f && std::abs(obstacle_center[1] - drone_position[1]) <= 2.5f)
                {
                    new_detections.push_back({-1, obstacle_center, obstacle_radius, 0});
                }
            }
        }

        // 2. 尝试将新检测到的障碍物与历史障碍物匹配
        for (size_t i = 0; i < new_detections.size(); ++i)
        {
            Eigen::Vector2f new_center = new_detections[i].position;
            float new_radius = new_detections[i].radius;
            bool matched_new = false;

            // 查找最近的历史障碍物
            int best_match_idx = -1;
            float min_distance = std::numeric_limits<float>::max();

            // 检查historical_obstacle_matched是否与historical_obstacles大小一致
            if (historical_obstacle_matched.size() != historical_obstacles.size())
            {
                ROS_ERROR("Size mismatch! historical_obstacle_matched: %zu, historical_obstacles: %zu",
                          historical_obstacle_matched.size(), historical_obstacles.size());
                historical_obstacle_matched.resize(historical_obstacles.size(), false);
            }

            for (size_t j = 0; j < historical_obstacles.size(); ++j)
            {
                float distance = (historical_obstacles[j].position - new_center).norm();
                if (distance < min_distance && distance <= matching_distance_threshold)
                {
                    min_distance = distance;
                    best_match_idx = j;
                }
            }

            // 如果找到匹配的历史障碍物
            if (best_match_idx != -1)
            {
                // 确保索引有效
                if (best_match_idx >= 0 && best_match_idx < historical_obstacle_matched.size() &&
                    best_match_idx < historical_obstacles.size())
                {
                    float alpha = 0.7f;
                    historical_obstacles[best_match_idx].position =
                        alpha * new_center + (1.0f - alpha) * historical_obstacles[best_match_idx].position;
                    historical_obstacles[best_match_idx].radius =
                        alpha * new_radius + (1.0f - alpha) * historical_obstacles[best_match_idx].radius;
                    historical_obstacles[best_match_idx].missed_count = 0;
                    historical_obstacle_matched[best_match_idx] = true;
                    matched_new = true;
                }
                else
                {
                    ROS_ERROR("Invalid best_match_idx: %d", best_match_idx);
                }
            }

            // 如果没有匹配到历史障碍物，创建新的
            if (!matched_new)
            {
                historical_obstacles.push_back({
                    next_id++,
                    new_center,
                    new_radius,
                    0 // missed_count
                });
                // 新的障碍物也需要标记为已匹配
                historical_obstacle_matched.push_back(true);
            }
        }

        // 3. 更新未匹配的历史障碍物的missed_count
        for (size_t j = 0; j < historical_obstacles.size(); ++j)
        {
            // 关键：检查索引是否有效
            if (j < historical_obstacle_matched.size())
            {
                if (!historical_obstacle_matched[j])
                {
                    historical_obstacles[j].missed_count++;
                }
            }
            else
            {
                // 如果标记数组大小不匹配，重新调整
                ROS_WARN("historical_obstacle_matched size mismatch, adjusting");
                historical_obstacle_matched.resize(historical_obstacles.size(), false);
            }
        }

        // 4. 删除连续未检测到超过阈值的障碍物
        auto it = historical_obstacles.begin();
        while (it != historical_obstacles.end())
        {
            if (it->missed_count > max_missed_frames)
            {
                it = historical_obstacles.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // 5. 将剩余的历史障碍物作为当前帧的输出
        int output_id = 0;
        for (auto &obs : historical_obstacles)
        {
            obstacles.push_back({output_id++, obs.position, obs.radius, obs.missed_count});
        }
    }

    void logTheInfo(const livox_ros_driver::CustomMsg::ConstPtr &msg)
    {
        static int log_count = 0;
        if (if_debug == 1 && ++log_count % 10 == 0)
        {
            ROS_INFO("Drone: (%.2f, %.2f), Target: (%.2f, %.2f)",
                     local_pos.pose.pose.position.x, local_pos.pose.pose.position.y,
                     target_x, target_y);
            ROS_INFO("总共%d，接收：%lu，体素：%lu，密集：%lu，簇数量：%lu", msg->point_num, cloud_world->size(), cloud_voxel->size(), cloud_dense->size(), clusters.size());
            ROS_INFO("=== 障碍物列表 (共 %zu 个) ===", clusters.size());
            for (size_t i = 0; i < clusters.size(); ++i)
            {
                float cx = 0, cy = 0;
                for (const auto &pt : clusters[i]->points)
                {
                    cx += pt.x;
                    cy += pt.y;
                }
                cx /= clusters[i]->size();
                cy /= clusters[i]->size();
                ROS_INFO("  簇 %zu: 中心(%.2f, %.2f), 点数=%zu",
                         i, cx, cy, clusters[i]->size());
            }
            ROS_INFO("=== 最终障碍物列表 (共 %zu 个) ===", obstacles.size());
            for (size_t i = 0; i < obstacles.size(); ++i)
            {
                const auto &obs = obstacles[i];
                ROS_INFO("  障碍物 %d: 位置(%.2f, %.2f), 半径=%.2fm",
                         obs.id,
                         obs.position.x(),
                         obs.position.y(),
                         obs.radius);
                log_count = 0;
            }
        }
    }
};

std::vector<Obstacle> detect_obs::historical_obstacles;
int detect_obs::next_id{0};
// 全局包装函数：暴露给外部的cb入口（看起来像全局函数）
void livox_cb_wrapper(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    // 调用单例的成员函数
    detect_obs::getInstance().livox_custom_cb(msg);
}

#endif