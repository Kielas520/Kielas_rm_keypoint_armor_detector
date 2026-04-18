# demo/src/__init__.py

# 1. 导入核心类
from .detector import Detector

# 2. 定义对外接口
# 我们将 Detector 视为最高层接口，InferenceEngine 视为中层接口
__all__ = [
    'Detector'
]