import cv2
import os, sys
import shutil
import numpy as np
from pathlib import Path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from ultralytics import YOLO

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

# Load a model
model = YOLO('E:/Code/ultralytics/model/yolov8m-seg.pt')  # load an official model
# model = YOLO('E:/Code/ultralytics/model/yolov8x.pt')  # load an official model

file_dir = "F:/Train_Video/ADAS_Train_Video/ADSA_TDX_20240320_0327_train/car_collide_3_ljr/IMAGE_car_collide_3_ljr/Truck_Day_images"
for files in Path(file_dir).rglob('*.*g'):
    # Predict with the model
    results = model(files, conf=0.5)  # predict on an image

    # Process results list
    for result in results:
        orig_img = result.orig_img
        boxes = result.boxes.data.tolist()
        h, w = result.orig_shape
        names = result.names
        masks = result.masks
        if masks is None:
            # shutil.move(str(files), "E:/dataset/VideoData/Head_Top/1280x720_img/test_no_mask")
            continue
        for i, mask in enumerate(masks.data):
            
            mask = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask, (w, h))

            label = int(boxes[i][5])
            color = np.array(random_color(label))

            colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
            masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)

            mask_indices = mask_resized == 1
            orig_img[mask_indices] = (orig_img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)

        for i, points in enumerate(masks.xy):
            label = int(boxes[i][5])
            color = random_color(label)
            points = np.array(points, np.int32)
            cv2.drawContours(orig_img, [points], -1, color, 2)

        for obj in boxes:
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            cv2.rectangle(orig_img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
            caption = f"{names[label]} {confidence:.2f}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
            cv2.rectangle(orig_img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv2.putText(orig_img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

    cv2.imshow('result', orig_img)
    cv2.waitKey(0)