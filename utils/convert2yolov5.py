import os
import cv2
import shutil 
import random
import yaml

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
    
    if text: 
        with open(yolov5_txt_path, 'w') as f: 
            f.write(text.strip())
        return True
    return False


def create_yolov5_data(
    image_dir, 
    target_dir, 
    image_suffix, 
    target_suffix, 
    cls_names, 
    test_ratio=0.2, seed=10, digit=5
): 
    # create directories
    os.mkdir('./custom_data/')
    os.mkdir('./custom_data/images/')
    os.mkdir('./custom_data/labels/')
    for root in ['./custom_data/images/', './custom_data/labels/']: 
        os.mkdir(os.path.join(root, 'train'))
        os.mkdir(os.path.join(root, 'val'))
    
    # collect images
    image_names = []
    for image_name in [image_name for image_name in os.listdir(image_dir) if image_name.endswith(image_suffix)]:
        if os.path.isfile(os.path.join(target_dir, image_name.split('.')[0] + target_suffix)): 
            image_names.append(image_name)
    
    # do train val split
    image_names.sort()
    random.seed(seed)
    random.shuffle(image_names)
    
    test_size = int(len(image_names) * test_ratio)
    train_image_names = image_names[test_size:]
    test_image_names = image_names[:test_size]

    print(f'training set has {len(train_image_names)} images!')
    print(f'validation set has {len(test_image_names)} images!')
    
    # cp image & yolo_txt
    for mode in ['train', 'val']:
        if mode == 'train': 
            image_names = train_image_names
        elif mode == 'val': 
            image_names = test_image_names

        for image_name in image_names: 
            shutil.copyfile(
                os.path.join(image_dir, image_name), 
                os.path.join('./custom_data/images/', mode, image_name)
            )

            res = txt_to_yolov5_format(
                os.path.join(target_dir, image_name.split('.')[0] + target_suffix), 
                os.path.join('./custom_data/labels/', mode, image_name.split('.')[0] + '.txt'), 
                os.path.join(image_dir, image_name), 
                digit=5
            )

            if not res: 
                print(f'{image_name} has no object!')
            
    # create yaml file
    yaml_data = {
        'path': './custom_data/images/',
        'train': '/train', 
        'val': '/val',
        'nc': len(cls_names), 
        'names': {
            i+1: cls_names[i] for i in range(len(cls_names))
        }
    }
    
    with open('./custom_data.yaml', 'w') as f:
        yaml.dump(yaml_data, f, Dumper=yaml.CDumper)