import argparse
from config.prepare_config import *
from prepare.convert2voc import create_voc_data
from prepare.convert2yolov5 import create_yolov5_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLO data preparation!")
    parser.add_argument('--task', default='yolox')
    parser.add_argument('--image_dir', default='/content/train')
    parser.add_argument('--target_dir', default='/content/train')
    parser.add_argument('--image_suffix', default='.png')
    parser.add_argument('--target_suffix', default='.txt')
    args = parser.parse_args()

    assert args.task.lower() in ('yolox', 'yolov5', 'yolov6')

    root_dict = {
        'yolox': yolox_root,
        'yolov5': yolov5_root,
        'yolov6': yolov6_root
    }

    data_format_dict = {
        'yolox': 'voc',
        'yolov5': 'yolov5',
        'yolov6': 'voc'
    }


    if data_format_dict[args.task.lower()] == 'voc': 
        create_voc_data(
            root_dict[args.task.lower()],
            image_dir=args.image_dir,
            target_dir=args.target_dir,
            image_suffix=args.image_suffix,
            target_suffix=args.target_suffix,
            cls_dict={str(i):cls for i, cls in enumerate(cls_names)},
            data_ratio=(train_ratio, val_ratio), 
            seed = seed
        )

    elif data_format_dict[args.task.lower()] == 'yolov5': 
        create_yolov5_data(
            root_dict[args.task.lower()],
            image_dir=args.image_dir,
            target_dir=args.target_dir,
            image_suffix=args.image_suffix,
            target_suffix=args.target_suffix,
            cls_names=cls_names,
            data_ratio=(train_ratio, val_ratio), 
            seed = seed
        ) 