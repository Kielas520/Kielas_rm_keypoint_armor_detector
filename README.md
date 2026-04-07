# Kielas_rm_detector_train
手搓适合 rm 视觉体质的 yolo pose 模型来识别装甲板
ros2 bag record -o my_debug_data /image_raw /detector/armors_debug_info
ros2 bag record -o infantry_red_data /image_raw /detector/armors_debug_info


# 激活 ROS 环境及自定义消息
source /opt/ros/humble/setup.bash
source ~/DT46_V/install/setup.bash

# 运行脚本
python3 ~/Kielas_rm_detector_train/extract_ros2_bag/extract.py
