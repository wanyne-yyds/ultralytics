import os
import cv2
import sys
import json
import numpy as np
import supervision as sv
from pathlib import Path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from ultralytics import YOLO

def save_boxes_to_labelme_json(image_path, image_heighe, image_width, boxes, labels, save_path):
    labelme_data = {
        "version": "6.2.6",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_heighe,
        "imageWidth": image_width
    }

    for index, box in enumerate(boxes):
        label = labels[index].split(":")[0]
        shape = {
            "label": label,
            "points": [[int(box[0]), int(box[1])], [int(box[2]), int(box[3])]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)
    with open(save_path, "w") as json_file:
        json.dump(labelme_data, json_file, indent=4)

model = YOLO("E:/Code/ultralytics/runs/detect/ADAS3/weights/best.pt")
input_dir = r"F:\Train_Video\ADAS_Train_Video\ADSA_TDX_20240320_0327_train\car_collide_zlj\IMAGE_car_collide_zlj_night"
selected_classes = [0, 1, 2]

for image_file in Path(input_dir).rglob("*.*g"):
    # if "Day" not in image_file.parent.name:
    #     continue
    image = cv2.imread(str(image_file))
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.6]
    detections = detections[np.isin(detections.class_id, selected_classes)]

    #* 保存 Json 标签文件
    height, width = image.shape[:2]
    bboxes = detections.xyxy
    if len(bboxes) == 0:
        continue
    imgname = image_file.name
    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    save_boxes_to_labelme_json(imgname, height, width, bboxes, labels, image_file.with_suffix(".json"))

    # bounding_box_annotator = sv.BoundingBoxAnnotator()
    # label_annotator = sv.LabelAnnotator()


    # annotated_image = bounding_box_annotator.annotate(
    #     scene=image, detections=detections)
    # annotated_image = label_annotator.annotate(
    #     scene=annotated_image, detections=detections, labels=labels)
    
    # sv.plot_image(annotated_image, size=(10, 7))
