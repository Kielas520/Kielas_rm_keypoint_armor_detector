import os
import cv2
import shutil
import bisect
import subprocess
import rosbag2_py
from concurrent.futures import ProcessPoolExecutor
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import sys
from pathlib import Path
import multiprocessing
import threading
import yaml
import numpy as np

# 引入 rich 相关组件
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn
)
from rich.console import Console

console = Console()

def source_env(script_path):
    """加载 ROS2 环境"""
    script_path = Path(script_path).expanduser()
    if not script_path.exists(): return
    try:
        command = f"bash -c 'source {script_path} && env'"
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, text=True)
        for line in proc.stdout:
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value.strip()
                if key == "PYTHONPATH":
                    for path in value.strip().split(":"):
                        if path and path not in sys.path: sys.path.insert(0, path)
        proc.communicate()
    except: pass

class RosBagExtractor:
    def __init__(self):
        self.folder_map = {
            "hero_blue_data": 0, "infantry_blue_data": 2, "sentry_blue_data": 5,
            "hero_red_data": 6, "infantry_red_data": 8, "sentry_red_data": 11
        }
        self.root_dir = Path(__file__).resolve().parent.parent
        self.original_dir = self.root_dir / "extract_ros2_bag" / "original"
        self.raw_data_dir = self.root_dir / "data" / "raw"

    @staticmethod
    def process_single_bag(folder_name, target_id, progress_queue, task_id, diff_tol, original_dir, raw_data_dir):
        """自适应延迟计算与数据提取"""
        os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = ''
        os.environ['RCL_LOG_LEVEL'] = '40' 
        from rm_interfaces.msg import ArmorsDebugMsg
        
        bag_path = original_dir / folder_name
        if not bag_path.exists():
            progress_queue.put({"task_id": task_id, "type": "description", "value": "[red]目录缺失"})
            progress_queue.put({"task_id": task_id, "type": "done"})
            return

        # 准备目录
        base_id_dir = raw_data_dir / str(target_id)
        if base_id_dir.exists(): shutil.rmtree(base_id_dir)
        base_id_dir.mkdir(parents=True, exist_ok=True)
        photo_dir, label_dir = base_id_dir / "photos", base_id_dir / "labels"
        photo_dir.mkdir(); label_dir.mkdir()

        bridge = CvBridge()
        storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        try:
            # 1. 预分析：获取图像总数 + 自适应计算延迟
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            
            # 获取图像总数
            metadata = reader.get_metadata()
            total_images = next((t.message_count for t in metadata.topics_with_message_count if t.topic_metadata.name == '/image_raw'), 0)
            progress_queue.put({"task_id": task_id, "type": "total", "value": total_images})
            progress_queue.put({"task_id": task_id, "type": "description", "value": "正在计算自适应延迟..."})

            # 2. 读取标签并分析延迟分布
            label_data = []
            latencies = []
            reader.set_filter(rosbag2_py.StorageFilter(topics=['/detector/armors_debug_info']))
            
            while reader.has_next():
                (topic, data, recv_t) = reader.read_next()
                msg = deserialize_message(data, ArmorsDebugMsg)
                header_t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                
                # 计算该帧的延迟：录制时刻 - 逻辑时刻
                # 录制时刻包含：推理时间 + 传输时间
                latencies.append(recv_t - header_t)
                
                armors = [{'id': a.armor_id, 'color': a.color, 'pts': [a.l_light_up_dx, a.l_light_up_dy, a.l_light_down_dx, a.l_light_down_dy, a.r_light_up_dx, a.r_light_up_dy, a.r_light_down_dx, a.r_light_down_dy]} for a in msg.armors_debug]
                label_data.append({"header_t": header_t, "armors": armors})

            if not label_data:
                progress_queue.put({"task_id": task_id, "type": "description", "value": "[yellow]无标签消息"})
                progress_queue.put({"task_id": task_id, "type": "done"})
                return

            # 使用中位数平滑掉系统抖动产生的延迟偏移
            adaptive_latency_ns = int(np.median(latencies))
            # 注意：如果算出来是负数，说明录制时刻不可信，或者 header 有问题，回退到 0
            adaptive_latency_ns = max(0, adaptive_latency_ns)
            
            # 补偿后的标签时间戳序列（尝试对应回图像曝光时间）
            comp_label_ts = [x["header_t"] - adaptive_latency_ns for x in label_data]
            
            desc_info = f"延迟: {adaptive_latency_ns/1e6:.1f}ms"
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"{desc_info} | 导出中..."})

            # 3. 匹配图像
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            reader.set_filter(rosbag2_py.StorageFilter(topics=['/image_raw']))
            
            img_count, batch = 0, 0
            while reader.has_next():
                (_, data, _) = reader.read_next()
                msg = deserialize_message(data, Image)
                img_t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                
                # 寻找最接近补偿后标签的图像
                idx = bisect.bisect_left(comp_label_ts, img_t)
                best_idx = -1
                min_diff = float('inf')
                for i in [idx-1, idx]:
                    if 0 <= i < len(comp_label_ts):
                        diff = abs(img_t - comp_label_ts[i])
                        if diff < min_diff: min_diff, best_idx = diff, i

                if best_idx != -1 and min_diff < diff_tol:
                    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                    name = f"{img_count:06d}"
                    cv2.imwrite(str(photo_dir / f"{name}.jpg"), cv_img)
                    with open(label_dir / f"{name}.txt", 'w') as f:
                        for a in label_data[best_idx]["armors"]:
                            f.write(f"{a['id']} {a['color']} {' '.join(map(str, a['pts']))}\n")
                    img_count += 1
                
                batch += 1
                if batch >= 20:
                    progress_queue.put({"task_id": task_id, "type": "advance", "value": batch})
                    batch = 0

            progress_queue.put({"task_id": task_id, "type": "advance", "value": batch})
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[green]完成 ({img_count}帧)"})
            progress_queue.put({"task_id": task_id, "type": "done"})
        except Exception as e:
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[red]异常: {str(e)[:20]}"})
            progress_queue.put({"task_id": task_id, "type": "done"})

    def extract(self, diff_tol):
        console.print(f"[bold blue]ROS2 数据自适应对齐提取器[/bold blue]")
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[name]}"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            refresh_per_second=5
        )

        def update_ui():
            while True:
                msg = progress_queue.get()
                if msg is None: break
                tid = msg['task_id']
                if msg['type'] == 'total': progress.update(tid, total=msg['value'])
                elif msg['type'] == 'advance': progress.advance(tid, advance=msg['value'])
                elif msg['type'] == 'description': progress.update(tid, description=msg['value'])
                elif msg['type'] == 'done': progress.update(tid, completed=progress.tasks[tid].total or 1)

        with progress:
            task_ids = {name: progress.add_task("初始化...", name=f"{name:<20}", total=None) for name in self.folder_map.keys()}
            ui_thread = threading.Thread(target=update_ui, daemon=True); ui_thread.start()

            # 使用进程池，注意 max_workers 根据 CPU 核心调整
            with ProcessPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(self.process_single_bag, n, tid, progress_queue, task_ids[n], diff_tol, self.original_dir, self.raw_data_dir) for n, tid in self.folder_map.items()]
                for f in futures: f.result()

            progress_queue.put(None); ui_thread.join()

if __name__ == '__main__':
    source_path = "~/DT46_V/install/setup.bash"
    config_path = "config.yaml"
    diff_tol = 15_000_000 # 容差设为 15ms，匹配更严格一点

    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            diff_tol = cfg.get("diff_tol_ns", diff_tol)

    if "DT46_V" not in os.environ.get("LD_LIBRARY_PATH", ""):
        source_env(source_path)
        os.execv(sys.executable, ['python3'] + sys.argv)

    RosBagExtractor().extract(diff_tol)