import platform
from ultralytics import YOLO
from ultralytics.data.converter import convert_json_to_yolo_obb

if __name__ == '__main__':

    # Convert json to yolo format
    # Json_file = r'D:/dataset/Motorcycle_Stop_Line_Dataset/Src/Labels/Train/Forgery'
    # convert_json_to_yolo_obb(Json_file) 

    # Load a model
    if platform.system().lower() == 'linux':
        model = YOLO('/home/ultralytics/ultralytics/cfg/models/v8/yolov8m.yaml').load(
            '/home/ultralytics/runs/detect/ADAS3/weights/best.pt')
        
    elif platform.system().lower() == 'windows':
        model = YOLO('E:/Code/ultralytics/ultralytics/cfg/models/v8/yolov8-obb.yaml').load(
            'E:/Code/ultralytics/runs/obb/Forgery_OBB2/weights/best.pt')
        
    else:
        raise NotImplementedError
    
    # # Train the model
    # results = model.train(data='/home/ultralytics/ultralytics/cfg/datasets/MED.yaml', 
    #                       epochs=160, 
    #                       imgsz=480,
    #                       batch=10,
    #                       workers=16,
    #                       name='Temp_OBB',
    #                       device=[1])

    # Predict on an image
    results = model(r'Z:\dataset\Motorcycle_Stop_Line_Dataset\Src\Labels\Train\Forgery\BuleWhite_Train_forgery', 
                    mode='predict',
                    save=True,
                    show_labels=True, 
                    conf=0.5,
                    # iou=0.7,
                    )  # predict on an image

    # Validate the model
    # metrics = model.val(data='/home/ultralytics/ultralytics/cfg/datasets/MED.yaml')  # no arguments needed, dataset and settings remembered

    # Export onnx
    # model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=480)