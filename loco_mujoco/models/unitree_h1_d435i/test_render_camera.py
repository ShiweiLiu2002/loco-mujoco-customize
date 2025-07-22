import mujoco
import mujoco.viewer
import numpy as np
import glfw
import cv2

# 模型路径和设置
MODEL_PATH = "./h1.xml"   # 替换为你自己的模型路径
CAMERA_NAME = "d435i_cam"
RESOLUTION = (640, 640)

# 初始化 OpenGL 上下文用于离屏渲染
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
offscreen_window = glfw.create_window(*RESOLUTION, "offscreen", None, None)
glfw.make_context_current(offscreen_window)

# 加载模型并初始化数据
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 设置摄像头（固定 camera）
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
if cam_id == -1:
    raise RuntimeError(f"Camera '{CAMERA_NAME}' not found.")

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
camera.fixedcamid = cam_id

# 离屏渲染上下文
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
viewport = mujoco.MjrRect(0, 0, *RESOLUTION)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 启动 MuJoCo 的 viewer 窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer launched. Press ESC in OpenCV window to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # 更新 MuJoCo viewer
        viewer.sync()

        # 离屏渲染 camera 视角
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        # 读取图像并显示（OpenCV）
        rgb = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera View", bgr)

        if cv2.waitKey(1) == 27:  # ESC
            break

# 清理资源
cv2.destroyAllWindows()
glfw.terminate()
del context
del scene
