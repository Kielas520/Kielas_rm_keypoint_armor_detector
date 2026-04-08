import os
import sys
import cv2
import shutil
import bisect
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn
)

def source_env(script_path: Path):
    script_path = script_path.expanduser()
    if not script_path.exists():
        print(f"[跳过] 未找到环境脚本: {script_path}")
        return

    try:
        command = f"bash -c 'source {script_path} && env'"
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, text=True)

        for line in proc.stdout:
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip()
                os.environ[key] = value

                if key == "PYTHONPATH":
                    for p in value.split(":"):
                        if p and p not in sys.path:
                            sys.path.insert(0, p)

        proc.communicate()
        print("[环境] 成功 Source 且已更新 sys.path")
    except Exception as e:
        print(f"[环境] 加载环境失败: {e}")

class RosBagExtractor:
    def __init__(self):
        # 目标 ID 映射
        self.folder_map = {
            "hero_blue_data": 0,
            "infantry_blue_data": 2,
            "sentry_blue_data": 5,
            "hero_red_data": 6,
            "infantry_red_data": 8,
            "sentry_red_data": 11
        }
        
        self.root_dir = Path(__file__).resolve().parent.parent
        self.original_dir = self.root_dir / "extract_ros2_bag" / "original"
        self.raw_data_dir = self.root_dir / "data" / "raw"

    def prepare_directory(self, target_path: Path):
        try:
            if target_path.exists():
                shutil.rmtree(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass 

    def get_reader(self, path: Path, topics: list = None):
        """
        获取 Reader 并应用 Topic 过滤，这是加速的核心。
        """
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        if topics:
            storage_filter = rosbag2_py.StorageFilter(topics=topics)
            reader.set_filter(storage_filter)
            
        return reader

    def process_single_bag(self, folder_name: str, target_id: int, progress: Progress, task_id):
        progress.update(task_id, description=f"[yellow]等待读取: {folder_name}")
        
        bag_path = self.original_dir / folder_name
        base_id_dir = self.raw_data_dir / str(target_id)
        self.prepare_directory(base_id_dir)

        photo_dir = base_id_dir / "photos"
        label_dir = base_id_dir / "labels"
        photo_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 预读取元数据以精确控制进度条
            metadata = rosbag2_py.Info().read_metadata(str(bag_path), 'sqlite3')
            label_msg_count = 0
            image_msg_count = 0
            
            for topic_info in metadata.topics_with_message_count:
                name = topic_info.topic_metadata.name
                count = topic_info.message_count
                if name == '/detector/armors_debug_info':
                    label_msg_count = count
                elif name == '/image_raw':
                    image_msg_count = count
            
            total_work = label_msg_count + image_msg_count
            if total_work == 0:
                progress.remove_task(task_id)
                return

            progress.start_task(task_id)
            progress.update(task_id, total=total_work, description=f"[cyan]正在解析: {folder_name}")

            bridge = CvBridge()
            label_indices = []
            
            # 第一遍：仅提取标签（由于跳过了图像 Blob，速度极快）
            label_reader = self.get_reader(bag_path, topics=['/detector/armors_debug_info'])
            while label_reader.has_next():
                (topic, data, t) = label_reader.read_next()
                msg = deserialize_message(data, ArmorsDebugMsg)
                armor_list = []
                for a in msg.armors_debug:
                    armor_list.append({
                        'id': a.armor_id, 'color': a.color,
                        'pts': [a.l_light_up_dx, a.l_light_up_dy, a.l_light_down_dx, a.l_light_down_dy,
                                a.r_light_up_dx, a.r_light_up_dy, a.r_light_down_dx, a.r_light_down_dy]
                    })
                label_indices.append((t, armor_list))
                progress.advance(task_id)

            if not label_indices:
                progress.console.print(f"[yellow][跳过] {folder_name} 未发现标签。[/yellow]")
                progress.remove_task(task_id)
                return

            label_indices.sort(key=lambda x: x[0])
            label_timestamps = [x[0] for x in label_indices]

            # 第二遍：提取图像并匹配
            img_reader = self.get_reader(bag_path, topics=['/image_raw'])
            img_count = 0
            while img_reader.has_next():
                (topic, data, t) = img_reader.read_next()
                
                # 时间戳二分查找匹配
                idx = bisect.bisect_left(label_timestamps, t)
                best_idx = idx
                if idx > 0 and (idx == len(label_timestamps) or abs(t - label_timestamps[idx-1]) < abs(t - label_timestamps[idx])):
                    best_idx = idx - 1

                # 50ms 阈值匹配
                if abs(t - label_timestamps[best_idx]) < 50_000_000:
                    msg = deserialize_message(data, Image)
                    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

                    file_name = f"{img_count:06d}"
                    cv2.imwrite(str(photo_dir / f"{file_name}.jpg"), cv_img)

                    matched_armors = label_indices[best_idx][1]
                    with open(label_dir / f"{file_name}.txt", 'w') as f:
                        for a in matched_armors:
                            pts_str = " ".join(map(str, a['pts']))
                            f.write(f"{a['id']} {a['color']} {pts_str}\n")
                    img_count += 1
                
                progress.advance(task_id)

            progress.update(task_id, description=f"[bold green]✓ {folder_name} ({img_count}张)")

        except Exception as e:
            progress.console.print(f"[red]错误: {folder_name} -> {e}[/red]")
            progress.remove_task(task_id)

    def extract(self):
        valid_tasks = []
        for folder_name, target_id in self.folder_map.items():
            bag_path = self.original_dir / folder_name
            if bag_path.exists():
                valid_tasks.append((folder_name, target_id))

        if not valid_tasks:
            print("未找到有效的数据包，请检查路径。")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            refresh_per_second=5,
        ) as progress:
            
            # 如果是 SSD，建议 max_workers 设为 4-6
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for folder_name, target_id in valid_tasks:
                    task_id = progress.add_task(f"[grey]等待调度: {folder_name}", start=False)
                    futures.append(
                        executor.submit(self.process_single_bag, folder_name, target_id, progress, task_id)
                    )
                
                for future in futures:
                    future.result()

if __name__ == '__main__':
    source_path = Path("~/DT46_V/install/setup.bash")

    # 动态 Source 环境并热重启进程
    if "DT46_V" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("环境不完全，正在 Source 并重启进程...")
        source_env(source_path)
        os.execv(sys.executable, ['python3'] + sys.argv)

    # 确保在环境加载后导入自定义消息类型
    try:
        from rm_interfaces.msg import ArmorsDebugMsg
    except ImportError:
        print("[错误] 无法导入 ArmorsDebugMsg，请检查 setup.bash 是否正确。")
        sys.exit(1)

    extractor = RosBagExtractor()
    extractor.extract()