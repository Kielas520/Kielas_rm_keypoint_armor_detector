import sys
import cv2
from tools.hik_camera.src.hik_camera import HikCamera

cap = HikCamera(0)

if not cap.open():
    print("❌ 无法打开相机")
    sys.exit()

# 设置初始曝光：例如 10000us (10ms)
cap.set_exposure(10000, auto_mode=False)

print(f"📷 当前曝光时间: {cap.get_exposure()} us")
print("按 'W' 增加曝光，'S' 减少曝光，'Q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break

    # 显示
    cv2.imshow('HikRobot Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        current = cap.get_exposure()
        cap.set_exposure(current + 1000) # 增加 1ms
    elif key == ord('s'):
        current = cap.get_exposure()
        cap.set_exposure(max(100, current - 1000)) # 减少 1ms

cap.close()
cv2.destroyAllWindows()