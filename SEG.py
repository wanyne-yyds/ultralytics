import os
from pathlib import Path

import cv2
import supervision as sv

from ultralytics import YOLO
from ultralytics.data.converter import convert_json_to_yolo_seg

if __name__ == "__main__":
    # convert json to yolo format
    # Json_file = r'/home/data/Data_trainset/YOLOv8_Seg_LaneDataset/Src/day/val/'
    # convert_json_to_yolo_seg(Json_file)

    # Load a model
    model = YOLO("/home/ultralytics/ultralytics/cfg/models/v8/yolov8n-seg.yaml").load(
        "/home/ultralytics/runs/segment/Lane_seg_yolov8n_day/weights/best.pt"
    )  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(
        data="/home/ultralytics/ultralytics/cfg/datasets/lane-seg.yaml",
        epochs=160,
        imgsz=480,
        batch=258,
        workers=16,
        name="Lane_seg_yolov8n_day",
        device=[2, 1],
    )

    # export onnx
    # model = YOLO('/home/ultralytics/ultralytics/cfg/models/v8/yolov8n-seg.yaml').load(
    #     '/home/ultralytics/runs/segment/Lane_seg_yolov8n_day/weights/best.pt'
    # )
    # model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=480)

    # Predict Video
    model = YOLO("/home/ultralytics/runs/segment/Lane_seg_yolov8n_day/weights/best.pt")
    model_claees = {0: "lane"}
    input = "/home/ultralytics/lane-test-video"
    save_path = "/home/ultralytics/lane-test-video_draw"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for videofile in Path(input).rglob("*.mp4"):
        out_video = os.path.join(save_path, videofile.name)
        video_info = sv.VideoInfo.from_video_path(str(videofile))
        frames_generator = sv.get_video_frames_generator(str(videofile))

        with sv.VideoSink(target_path=out_video, video_info=video_info) as sink:
            for i, frame in enumerate(frames_generator):
                results = model(frame, conf=0.5)[0]
                detections = sv.Detections.from_ultralytics(results)
                if len(detections.xyxy) == 0:
                    sink.write_frame(frame)
                    continue
                mask_annotator = sv.MaskAnnotator()
                label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)
                labels = [
                    "%s: %.2f" % (model_claees[class_id], confidence)
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]

                annotated_image = mask_annotator.annotate(scene=frame, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

                sink.write_frame(annotated_image)
                print(f"{i} frame processed")
