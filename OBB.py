import platform
from ultralytics import YOLO
from ultralytics.data.converter import convert_json_to_yolo_obb

if __name__ == '__main__':

    # Convert json to yolo format
    # Json_file = r'D:/dataset/Motorcycle_Stop_Line_Dataset/Src/Labels/Train/Forgery'
    # convert_json_to_yolo_obb(Json_file) 

    # Load a model
    if platform.system().lower() == 'linux':
        #TODO 需要更换 Conv 激活函数 ->  nn.ReLU6()
        model = YOLO('/home/ultralytics/ultralytics/cfg/models/v8/yolov8s-obb-mobilenetv2.yaml').load(
            '/home/ultralytics/runs/obb/MobilenetV2_OBB_SPPF5/weights/last.pt'
        )
        
        #TODO 需要更换 Conv 激活函数 ->  nn.SiLU()
        # model = YOLO('/home/ultralytics/ultralytics/cfg/models/v8/yolov8n-obb.yaml').load(
        #     '/home/ultralytics/runs/obb/MEOBB_02/weights/last.pt')

    elif platform.system().lower() == 'windows':
        #TODO 需要更换 Conv 激活函数 ->  nn.ReLU6()
        model = YOLO('E:/Code/ultralytics/ultralytics/cfg/models/v8/yolov8s-obb-mobilenetv2.yaml').load(
            'E:/Code/ultralytics/runs/obb/MobilenetV2_OBB2/weights/best.pt')

        #TODO 需要更换 Conv 激活函数 ->  nn.SiLU()
        # model = YOLO('E:/Code/ultralytics/ultralytics/cfg/models/v8/yolov8n-obb.yaml').load(
        #     'E:/Code/ultralytics/runs/obb/MEOBB_02/weights/best.pt')

    else:
        raise NotImplementedError
    
    # Train the model
    # results = model.train(data='/home/ultralytics/ultralytics/cfg/datasets/MED.yaml', 
    #                       epochs=160, 
    #                       imgsz=480,
    #                       batch=258,
    #                       workers=32,
    #                       name='MobilenetV2_OBB_SPPF5',
    #                       device=[3, 2, 1])

    # Predict on an image
    # results = model(r'Z:\dataset\Motorcycle_Stop_Line_Dataset\Src\Labels\Train\BuleWhite_Train', 
    #                 mode='predict',
    #                 save=True,
    #                 show_labels=True, 
    #                 conf=0.5,
    #                 # iou=0.7,
    #                 )

    # Validate the model
    metrics = model.val(data='/home/ultralytics/ultralytics/cfg/datasets/MED.yaml', 
                        imgsz=480,
                        conf=0.5,
                        )  # no arguments needed, dataset and settings remembered

    # Export onnx
    # model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=480)