## 環境
```python
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
!git clone https://github.com/travisergodic/object_detection_project.git
```

## 取得資料
```python
%cp "/content/drive/MyDrive/2-專案/無人飛機計數/Training Dataset_v3.zip" "/content/"
!unzip "/content/drive/MyDrive/2-專案/無人飛機計數/Training Dataset_v3.zip"  -d "/content/YOLOX/"
%cp "/content/drive/MyDrive/2-專案/無人飛機計數/Public Testing Dataset_v2.zip" "/content/"
!unzip "/content/Public Testing Dataset_v2.zip" -d "/content/YOLOX/"
```

## 建立 custom_data 資料夾
```python
%cd /content/yolov5/object_detection_project
from utils.convert2yolov5 import create_yolov5_data

cls_names = ['car', 'hov', 'person', 'motorcycle']

create_yolov5_data(
    '/content/train',
    '/content/train',
    '.png',
    '.txt',
    cls_names
)
```
將 `custom_data` 資料夾 & `custom_data.yaml` 放到 `yolov5/` 路徑下

## 訓練模型
```
%cd /content/yolov5
!python train.py --img 640 --batch 16 --epochs 3 --data custom_data.yaml --weights 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m_Objects365.pt'
```

