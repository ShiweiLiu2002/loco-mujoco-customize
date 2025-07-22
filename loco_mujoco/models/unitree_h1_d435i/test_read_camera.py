import mujoco
import mujoco.viewer
import numpy as np
import os

# 模型路径
MODEL_DIR = "./"
H1_PATH = os.path.join(MODEL_DIR, "h1.xml")
CAMERA_NAME = "d435i_cam"

# 加载模型
model = mujoco.MjModel.from_xml_path(H1_PATH)
data = mujoco.MjData(model)

# 初始化 Renderer，用于 camera 图像渲染
renderer = mujoco.Renderer(model, height=640, width=640)

# 获取 camera id
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
if cam_id == -1:
    raise RuntimeError(f"Camera '{CAMERA_NAME}' not found.")

# 启动 MuJoCo Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("MuJoCo Viewer launched. Running simulation for 100 steps...")

    for step in range(1000):
        # 执行一步模拟
        mujoco.mj_step(model, data)

        # 更新 viewer 画面
        viewer.sync()

        # 渲染 camera 图像（不显示、不保存，仅验证）
        renderer.update_scene(data, camera=cam_id)
        img = renderer.render()  # img.shape: (640, 640, 3)
        
        # 简单确认图像成功读取
        print(f"Step {step + 1}: Captured image with shape {img.shape}")
