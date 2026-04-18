# tools/hik_camera/src/__init__.py

import os
import sys
from pathlib import Path

# 1. 自动化路径处理：确保 hik_lib 和 MvImport 能被系统找到
BASE_DIR = Path(__file__).resolve().parent.parent # 指向 tools/hik_camera/
LIB_DIR = BASE_DIR / "hik_lib"
MV_IMPORT_PATH = BASE_DIR / "MvImport"

# 将 DLL 目录添加到系统搜索路径 (Windows 特有)
if LIB_DIR.exists():
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(str(LIB_DIR))
    os.environ['PATH'] = str(LIB_DIR) + os.pathsep + os.environ['PATH']

# 将 MvImport 放入 Python 搜索路径
if str(MV_IMPORT_PATH) not in sys.path:
    sys.path.append(str(MV_IMPORT_PATH))

# 2. 导出核心类
from .hik_camera import HikCamera

__all__ = ['HikCamera']