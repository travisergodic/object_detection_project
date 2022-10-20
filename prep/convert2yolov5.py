import os
import cv2
import shutil 
import random
import yaml
from tqdm import tqdm
from pathlib import Path

def txt_to_yolov5_format(raw_txt_path, yolov5_txt_path, image_path, digit=5): 
    image_height, image_width, _ = cv2.imread(image_path).shape
    
    text =''
    with open(raw_txt_path, 'r') as f: 
        for line in f.readlines(): 
            cls, x_min, y_min, w, h = [int(ele) for ele in line.split(',')]
            x_c, y_c = x_min + w/2, y_min + h/2
            x_c, y_c, w, h = x_c/image_width, y_c/image_height, w/image_width, h/image_height
            x_c, y_c, w, h = round(x_c, digit), round(y_c, digit), round(w, digit), round(w, digit)  
            text += ' '.join([str(cls), str(x_c), str(y_c), str(w), str(h)]) 
            text += '\n'
     
    with open(yolov5_txt_path, 'w') as f: 
        f.write(text.strip())
    
    if text:
        return True
    return False


def create_yolov5_data(
    root,
    image_dir, 
    target_dir, 
    image_suffix, 
    target_suffix, 
    cls_names, 
    data_ratio, 
    seed=10, digit=5
): 
    root = Path(root)
    # create directories
    os.mkdir(root / 'custom_data/')
    os.mkdir(root / 'custom_data/images/')
    os.mkdir(root / 'custom_data/labels/')
    for root in [root / 'custom_data/images/', root / 'custom_data/labels/']: 
        os.mkdir(root / 'train')
        os.mkdir(root / 'val')
    
    # collect images
    image_names = []
    for image_name in [image_name for image_name in os.listdir(image_dir) if image_name.endswith(image_suffix)]:
        if os.path.isfile(os.path.join(target_dir, image_name.split('.')[0] + target_suffix)): 
            image_names.append(image_name)
    
    # train val split
    image_names.sort()
    random.seed(seed)
    random.shuffle(image_names)
    
    train_ratio, val_ratio = data_ratio 
    train_size, val_size = int(train_ratio * len(image_names)), int(val_ratio * len(image_names)) 

    train_image_names = image_names[:train_size]
    test_image_names = image_names[train_size:train_size+val_size]

    print(f'training set has {len(train_image_names)} images!')
    print(f'validation set has {len(test_image_names)} images!')
    
    # cp image & yolo_txt
    for mode in ['train', 'val']:
        if mode == 'train': 
            image_names = train_image_names
        elif mode == 'val': 
            image_names = test_image_names

        for image_name in tqdm(image_names): 
            shutil.copyfile(
                os.path.join(image_dir, image_name), 
                os.path.join(root / 'custom_data/images/', mode, image_name)
            )

            res = txt_to_yolov5_format(
                os.path.join(target_dir, image_name.split('.')[0] + target_suffix), 
                os.path.join(root / 'custom_data/labels/', mode, image_name.split('.')[0] + '.txt'), 
                os.path.join(image_dir, image_name), 
                digit=digit
            )

            if not res: 
                print(f'{image_name} has no object!')
            
    # create yaml file
    yaml_data = {
        'train': root / 'custom_data/images/train', 
        'val': root / 'custom_data/images/val',
        'nc': len(cls_names), 
        'names': {
            i: cls_names[i] for i in range(len(cls_names))
        }
    }
    
    with open(root / 'custom_data.yaml', 'w') as f:
        yaml.dump(yaml_data, f, Dumper=yaml.CDumper)

if __name__ == '__main__': 
    create_yolov5_data(
        '/content/yolov5',
        '/content/train', 
        '/content/train', 
        '.png', 
        '.txt', 
        ['car', 'hov', 'person', 'motorcycle'], 
        data_ratio=(0.8, 0.2), 
        seed=10, digit=5
    ) 