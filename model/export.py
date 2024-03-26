# 切换至工程路径导入包
# add python path of this repo to sys.path
import os, sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from ultralytics import YOLO

# Load a model
model = YOLO(r"D:\code\ultralytics\model\yolov8m-seg.pt")  # load an official model
# model.train(data='D:/code/ultralytics/ultralytics/cfg/datasets/coco8.yaml', epochs=3)  # train head with three epochs')

# Export the model
model.export(format='onnx')
