from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO('/home/ultralytics/ultralytics/cfg/models/v8/yolov8m.yaml').load(
        '/home/ultralytics/runs/detect/ADAS3/weights/best.pt')
    # Train the model
    # results = model.train(data='/home/ultralytics/ultralytics/cfg/datasets/ADAS.yaml', 
    #                       epochs=100, 
    #                       imgsz=640,
    #                       batch=128,
    #                       workers=18,
    #                       name='ADAS',
    #                       device=[3, 2],
    #                       )

    # Predict on an image
    # results = model('/home/data/Data_trainset/YOLOv8_OBB_MotorcycleStopLineDataset/images/val/GreenT_test/3590_TV_CAM_shebei_20230802_101623.503.jpg', 
    #                 mode='predict',
    #                 save=True,
    #                 show_labels=True, 
    #                 conf=0.5,
    #                 # iou=0.7,
    #                 )  # predict on an image

    # Validate the model
    # Load a model
    # model = YOLO('/home/ultralytics/ultralytics/cfg/models/v8/yolov8n-obb.yaml').load(
    #     '/home/ultralytics/runs/obb/MEOBB_02/weights/best.pt'
    # )
    # Validate the model
    # metrics = model.val(data='/home/ultralytics/ultralytics/cfg/datasets/ADAS.yaml')  # no arguments needed, dataset and settings remembered

    # Export onnx
    model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)