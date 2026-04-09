import tarfile
import requests
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel

console = Console()

def download_rm_backgrounds(base_path="./background", limit=10000, force_refresh=True):
    # 1. 路径初始化 (使用 pathlib)
    root = Path(base_path).resolve()
    flag_file = root / "extract_finished.flag"
    tar_file = root / "val_256.tar"
    
    # 2. 强制刷新逻辑：如果发现 flag，清空文件夹重新开始
    if force_refresh and flag_file.exists():
        console.print(Panel("[bold red]检测到旧的完成标记，正在强制刷新文件夹...[/bold red]"))
        for item in root.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        console.print("[green]✔ 文件夹已清空。[/green]")
    elif not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    # Places365 验证集下载地址
    url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    
    # RM 赛场相关室内关键词 (优先级排序)
    rm_keywords = ['gymnasium', 'basketball_court', 'badminton_court', 'hangar', 'warehouse', 'garage']

    # 3. 带断点续传的下载逻辑
    if not flag_file.exists():
        headers = {}
        if tar_file.exists():
            resume_byte = tar_file.stat().st_size
            headers['Range'] = f'bytes={resume_byte}-'
            mode = 'ab'
            console.print(f"[yellow]续传模式：已完成 {resume_byte / 1024**2:.2f} MB，继续获取剩余部分...[/yellow]")
        else:
            resume_byte = 0
            mode = 'wb'

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=20)
            # 如果服务器返回 200 而不是 206，说明不支持续传，重头开始
            if response.status_code == 200:
                mode = 'wb'
                resume_byte = 0
            
            total_size = int(response.headers.get('content-length', 0)) + resume_byte

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task_id = progress.add_task("下载 Places365", total=total_size, completed=resume_byte)
                with tar_file.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
        except Exception as e:
            console.print(f"[bold red]下载中断:[/bold red] {e}\n[yellow]提示：再次运行脚本即可从断点处自动续传。[/yellow]")
            return

    # 4. 智能筛选与扁平化提取
    console.print(Panel(f"正在精准筛选 [bold yellow]RM 室内/体育馆风格背景[/bold yellow]\n目标目录: {root}", title="数据提取"))
    
    try:
        with tarfile.open(tar_file, 'r') as tar:
            console.print("[blue]正在扫描压缩包索引，请稍候...[/blue]")
            # 过滤出所有文件成员
            all_members = [m for m in tar.getmembers() if m.isfile()]
            
            # 筛选逻辑：优先 RM 相关，其次通用室内
            rm_related = [m for m in all_members if any(kw in m.name.lower() for kw in rm_keywords)]
            indoor_related = [m for m in all_members if 'indoor' in m.name.lower() and m not in rm_related]
            
            selected_members = (rm_related + indoor_related)[:limit]

            # Fallback 机制：如果没有匹配到任何关键词，直接提取前 limit 张
            if not selected_members:
                console.print("[bold red]警告：未通过关键词匹配到图片！[/bold red] 执行保底提取策略...")
                selected_members = all_members[:limit]
            else:
                console.print(f"[green]成功匹配：{len(rm_related)} 张核心赛场图，{len(selected_members) - len(rm_related)} 张通用室内图。[/green]")

            # 开始解压写入
            with Progress(console=console) as progress:
                task = progress.add_task("[cyan]正在写入图片...", total=len(selected_members))
                for member in selected_members:
                    # 强行提取文件名，去掉压缩包内的复杂路径结构
                    original_filename = Path(member.name).name
                    if original_filename:
                        member.name = original_filename
                        tar.extract(member, path=root)
                    progress.update(task, advance=1)
            
        # 5. 收尾工作
        flag_file.touch() # 创建完成标记
        if tar_file.exists():
            tar_file.unlink() # 删除 500MB+ 的压缩包，保持空间整洁
            
        console.print(f"[bold green]✔ 任务圆满完成！[/bold green]")
        console.print(f"最终图片存放于: [white]{root}[/white]")
        
    except Exception as e:
        console.print(f"[bold red]提取过程出错:[/bold red] {e}")

if __name__ == "__main__":
    # 配置信息：请确认此路径是你想要存放图片的路径
    PROJECT_BG_PATH = "/Users/kielas/project/Kielas_rm_detector_train/background"
    
    # 运行脚本
    # limit: 需要的图片张数
    # force_refresh: 如果设为 True，每次检测到 flag 会删掉旧图重新开始
    download_rm_backgrounds(base_path=PROJECT_BG_PATH, limit=10000, force_refresh=True)