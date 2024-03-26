from ultralytics import YOLO
from ultralytics.data.converter import convert_json_to_yolo_obb

if __name__ == '__main__':

    # convert json to yolo format
    Json_file = r'E:\dataset\Motorcycle_Stop_Line_Dataset\Src\Labels\Test'
    convert_json_to_yolo_obb(Json_file)

    # Load a model
    # model = YOLO('D:/code/ultralytics/model/yolov8n-obb.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n-obb.yaml')
    # Train the model
    # results = model.train(data='D:/code/ultralytics/ultralytics/cfg/datasets/dota8.yaml', epochs=10, imgsz=640)

    # results = model('E:/dataset/OpenSourceDataset/DOTA/dota8/images/train/P0011__1024__343___502.jpg', 
    #                 save=False,
    #                 show_labels=False)  # predict on an image

    # model.export(format='onnx')
