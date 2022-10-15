## 環境
```python
!git clone https://github.com/travisergodic/object_detection_project.git
%cd /content/YOLOX
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
!pip install -U pip && pip install -r requirements.txt
!pip install -v -e .
```

## 取得資料
```python
%cp "/content/drive/MyDrive/2-專案/無人飛機計數/Training Dataset_v3.zip" "/content/"
!unzip "/content/drive/MyDrive/2-專案/無人飛機計數/Training Dataset_v3.zip"  -d "/content/YOLOX/"
%cp "/content/drive/MyDrive/2-專案/無人飛機計數/Public Testing Dataset_v2.zip" "/content/"
!unzip "/content/Public Testing Dataset_v2.zip" -d "/content/YOLOX/"
```

## 建立 VOCdevkit
```python
%cd "/content/YOLOX/object_detection_project/"  
from utils.convert2voc import create_VOCdevkit, train_test_split
from utils.statistics import bboxes_statistics

cls_dict = {
    '0': 'car',
    '1' : 'hov',
    '2' : 'person',
    '3' : 'motorcycle',
}

create_VOCdevkit('/content/YOLOX/train', '/content/YOLOX/train', '.png', '.txt', cls_dict)
train_test_split('/content/YOLOX/VOCdevkit/VOC2007/ImageSets/Main/')
```
將 `VOCdevkit` 資料夾移動到 `/content/YOLOX/datasets/` 路徑下

## 下載與訓練權重檔
```
%cd /content/YOLOX
%mkdir /content/YOLOX/weights
!wget -O weights/yolox_s.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

# 訓練模型
1. 修改 `content/YOLOX/exps/example/yolox_voc/yolox_voc_s.py` 中的配置
  + `self.num_classes = {num_classes}`
  + `self.max_epoch = {max_epoch}`
  + `image_sets = [('2007', 'trainval')]`
2. 修改 `content/YOLOX/yolox/data/datasets/voc_classes.py` 中的類別
3. 修改 `content/YOLOX/yolox/data/datasets/voc.py` 中的圖片後綴
4. 執行訓練指令
   ```python
   %cd /content/YOLOX
   !python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 16 --fp16 -o -c weights/yolox_s.pth
   ```

## 結果展示
1. 修改 `content/YOLOX/yolox/data/datasets/coco_classes.py` 中的類別
2. 執行指令
```
%cd /content/YOLOX
!python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c /content/YOLOX/YOLOX_outputs/yolox_voc_s/last_epoch_ckpt.pth --path /content/YOLOX/datasets/VOCdevkit/VOC2007/JPEGImages/img0001.png --conf 0.25 --nms 0.5 --tsize 640 --save_result
```
  


