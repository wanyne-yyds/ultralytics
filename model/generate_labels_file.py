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

def create_shape(label, box):
    if len(box) != 4:
        points = [[box[i], box[i+1]] for i in range(0, len(box), 2)]
        shape_type = "polygon"
    else:
        points = [[box[0], box[1]], [box[2], box[3]]]
        shape_type = "rectangle"
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": shape_type,
        "flags": {}
    }

def save_boxes_to_labelme_json(image_path, image_heighe, image_width, boxes, labels, save_path, save_metrics_file=True):

    global MED_CLASSES
    if save_metrics_file:
        with open(save_path.with_suffix(".txt"), "w") as txt_file:
            for index, box in enumerate(boxes):
                label, score = labels[index].split(" ")
                label = MED_CLASSES[int(label)]
                box = box.flatten().astype(int).tolist()
                txt_file.write(f"{label} {score} {' '.join(map(str, box))}\n")
    else:
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
            label = MED_CLASSES[int(labels[index].split(" ")[0])]
            box = box.flatten().astype(int).tolist()
            shape = create_shape(label, box)
            labelme_data["shapes"].append(shape)
        with open(save_path, "w") as json_file:
            json.dump(labelme_data, json_file, indent=4)

if __name__ == '__main__':

    #TODO 需要更换 Conv 激活函数 ->  nn.ReLU6()
    model = YOLO('E:/Code/ultralytics/ultralytics/cfg/models/v8/yolov8s-obb-mobilenetv2.yaml').load(
        'E:/Code/ultralytics/runs/obb/MobilenetV2_OBB2/weights/last.pt')

    #TODO 需要更换 Conv 激活函数 ->  nn.SiLU()
    # model = YOLO('E:/Code/ultralytics/ultralytics/cfg/models/v8/yolov8n-obb.yaml').load(
    #     'E:/Code/ultralytics/runs/obb/MEOBB_02/weights/last.pt')

    input_dir = Path("Z:/dataset/Motorcycle_Stop_Line_Dataset/Src/Labels/Test")
    output_dir = Path(str(input_dir) + "_obb_pre_labels")

    # 过滤指定类别 Example: [0, 1, 2, 3]
    selected_classes = None
    # 过滤置信度 Example: 0.5        
    filter_confidence = 0.25
    # True: 保存 Txt 标签文件到 'output_dir' 目录; False: 保存 Json 标签文件到输入目录
    save_txt_file = True
    # True: 显示图片; False: 保存标签文件
    show_images = False

    MED_CLASSES = {0:'blue_white', 1:'red_white', 2:'green_white', 3:'greenT'}

    for image_file in input_dir.rglob("*.*g"):

        image = cv2.imread(str(image_file))
        results = model(image, imgsz=480, conf=0.001, iou=0.1)[0]
        detections = sv.Detections.from_ultralytics(results)

        if filter_confidence is not None:
            detections = detections[detections.confidence > filter_confidence]
        if selected_classes is not None:
            detections = detections[np.isin(detections.class_id, selected_classes)]

        labels = [
            f"{model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        if show_images:
            if hasattr(results, "obb"):
                oriented_box_annotator = sv.OrientedBoxAnnotator()
                annotated_image = oriented_box_annotator.annotate(
                    scene=image, detections=detections)
            else:
                bounding_box_annotator = sv.BoundingBoxAnnotator()
                annotated_image = bounding_box_annotator.annotate(
                    scene=image, detections=detections)
                
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)
            sv.plot_image(annotated_image, size=(10, 7))
        else:
            imgname = image_file.name
            height, width = image.shape[:2]

            if hasattr(results, "obb"):
                bboxes = detections.data['xyxyxyxy']
            else:
                bboxes = detections.xyxy

            output_son = image_file.with_suffix("json") if not save_txt_file else \
                Path(str(image_file).replace(str(input_dir), str(output_dir))).with_suffix(".txt")
            Path(output_son).parent.mkdir(parents=True, exist_ok=True)

            save_boxes_to_labelme_json(
            imgname, height, width, bboxes, labels, output_son, save_metrics_file=save_txt_file
            )