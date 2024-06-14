import platform
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    if platform.system().lower() == 'linux':
        model = YOLO('/home/ultralytics/ultralytics/cfg/models/v8/yolov8m.yaml').load(
            '/home/ultralytics/runs/detect/ADAS3/weights/best.pt')
        
    elif platform.system().lower() == 'windows':
        # model = YOLO('E:/Code/ultralytics/ultralytics/cfg/models/v8/yolov8m.yaml').load(
        #     'E:/Code/ultralytics/runs/detect/ADAS3/weights/best.pt')
        
        model = YOLO('E:/Code/ultralytics/model/yolov8x.pt')
    else:
        raise NotImplementedError
    
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
    # results = model('F:/Train_Video/ADAS_Train_Video/ADSA_TDX_20240320_0327_train/car_collide_2_cze/Bus/Day_images', 
    #                 mode='predict',
    #                 save=True,
    #                 show_labels=False, 
    #                 conf=0.5,
    #                 show=True,
    #                 # iou=0.7,
    #                 )  # predict on an image

    # Validate the model
    # Validate the model
    # metrics = model.val(data='/home/ultralytics/ultralytics/cfg/datasets/ADAS.yaml')  # no arguments needed, dataset and settings remembered

    # Export onnx
    # model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)

    # Track on a video
    results = model.track(r"Z:\dataset\VideoData\IPC\BSJ-Office\video(2).mp4", classes=[0], show=True)