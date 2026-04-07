import os
import cv2
import rosbag2_py
import bisect
from concurrent.futures import ThreadPoolExecutor
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

# 尝试导入自定义消息
try:
    from rm_interfaces.msg import ArmorsDebugMsg
except ImportError:
    print("错误：未找到 rm_interfaces 消息定义，请先执行 source install/setup.bash")

class RosBagExtractor:
    def __init__(self):
        self.bridge = CvBridge()
        # 文件夹与 ID 映射关系
        self.folder_map = {
            "hero_blue_data": 0,      # B1
            "infantry_blue_data": 2,  # B3
            "sentry_blue_data": 5,    # B7
            "hero_red_data": 6,       # R1
            "infantry_red_data": 8,   # R3
            "sentry_red_data": 11     # R7
        }
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.original_dir = os.path.join(self.root_dir, "extract_ros2_bag/original")
        self.raw_data_dir = os.path.join(self.root_dir, "data/raw")

    def get_reader(self, path):
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        return reader

    def process_single_bag(self, folder_name, target_id):
        """处理单个 Bag 的核心逻辑，供线程池调用"""
        bag_path = os.path.join(self.original_dir, folder_name)
        if not os.path.exists(bag_path):
            print(f"[跳过] 未找到目录: {bag_path}")
            return

        print(f"[开始] 正在处理: {folder_name} (ID: {target_id})")

        photo_dir = os.path.join(self.raw_data_dir, str(target_id), "photos")
        label_dir = os.path.join(self.raw_data_dir, str(target_id), "labels")
        os.makedirs(photo_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        try:
            # 第一步：预读取所有标签信息 [cite: 2]
            label_indices = []
            reader = self.get_reader(bag_path)
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                if topic == '/detector/armors_debug_info':
                    msg = deserialize_message(data, ArmorsDebugMsg)
                    armor_list = []
                    for a in msg.armors_debug: # [cite: 1, 2]
                        armor_list.append({
                            'id': a.armor_id, 'color': a.color,
                            'pts': [a.l_light_up_dx, a.l_light_up_dy, a.l_light_down_dx, a.l_light_down_dy,
                                    a.r_light_up_dx, a.r_light_up_dy, a.r_light_down_dx, a.r_light_down_dy]
                        })
                    label_indices.append((t, armor_list))

            if not label_indices:
                print(f"[警告] {folder_name} 中未找到标签数据。")
                return

            label_indices.sort(key=lambda x: x[0])
            label_timestamps = [x[0] for x in label_indices]

            # 第二步：流式读取图像并匹配保存
            reader = self.get_reader(bag_path)
            img_count = 0
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                if topic == '/image_raw':
                    idx = bisect.bisect_left(label_timestamps, t)
                    best_idx = idx
                    if idx > 0 and (idx == len(label_timestamps) or abs(t - label_timestamps[idx-1]) < abs(t - label_timestamps[idx])):
                        best_idx = idx - 1

                    if abs(t - label_timestamps[best_idx]) < 50_000_000:
                        msg = deserialize_message(data, Image)
                        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                        file_name = f"{img_count:06d}"
                        cv2.imwrite(os.path.join(photo_dir, f"{file_name}.jpg"), cv_img)

                        matched_armors = label_indices[best_idx][1]
                        with open(os.path.join(label_dir, f"{file_name}.txt"), 'w') as f:
                            for a in matched_armors:
                                pts_str = " ".join(map(str, a['pts']))
                                f.write(f"{a['id']} {a['color']} {pts_str}\n")

                        img_count += 1

            print(f"[完成] {folder_name}: 生成 {img_count} 组数据。")

        except Exception as e:
            print(f"[错误] 处理 {folder_name} 时发生异常: {e}")

    def extract(self):
        # 使用线程池并发执行，max_workers 建议设为 4-6
        print(f"准备并发处理 {len(self.folder_map)} 个数据包...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(self.process_single_bag, folder_name, target_id)
                for folder_name, target_id in self.folder_map.items()
            ]
            # 等待所有线程完成
            for future in futures:
                future.result()

if __name__ == '__main__':
    extractor = RosBagExtractor()
    extractor.extract()
