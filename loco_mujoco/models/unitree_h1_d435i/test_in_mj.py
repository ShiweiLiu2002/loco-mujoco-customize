from mujoco import viewer
import mujoco
model = mujoco.MjModel.from_xml_path("./h1.xml")
data = mujoco.MjData(model)
viewer.launch(model, data)