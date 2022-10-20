import os
import shutil
import random 
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil


def txt2xml(txt_path, xml_path, image_path, cls_dict):
    if (not os.path.isfile(txt_path)) and (not os.path.isfile(image_path)):
        print(f'both {txt_path}, {image_path} do not exist!')
        return False
    
    if not os.path.isfile(txt_path): 
        print(f'{txt_path} do not exist!')
        return False
        
    if not os.path.isfile(image_path): 
        print(f'{image_path} do not exist!')
        return False
        
    image_height, image_width, _ = cv2.imread(image_path).shape
    image_name = os.path.basename(image_path)
    
    with open(txt_path, 'r') as f: 
        s = f.read().strip()
        if s: 
            lines = s.split('\n')
        else: 
            print(f'{txt_path} has no object!')
            lines = []

    with open(xml_path, 'w') as f:
        f.write('<annotation>\n')
        f.write('\t<folder>data</folder>\n')
        f.write('\t<filename>' + image_name + '</filename>\n')
        f.write('\t<path>' + image_path + '</path>\n')
        f.write('\t<source>\n')
        f.write('\t\t<database>Unknown</database>\n')
        f.write('\t</source>\n')
        f.write('\t<size>\n')
        f.write('\t\t<width>' + str(image_width) + '</width>\n')
        f.write('\t\t<height>' + str(image_height) + '</height>\n')
        f.write('\t\t<depth>3</depth>\n') # assuming a 3 channel color image (RGB)
        f.write('\t</size>\n')
        f.write('\t<segmented>0</segmented>\n')

        for line in lines:
            cls, x, y, w, h = line.split(',')
            x_min, y_min = x, y
            x_max, y_max = str(int(x) + int(w)), str(int(y) + int(h))
            object_name = cls_dict[cls]
            # write each object to the file
            f.write('\t<object>\n')
            f.write('\t\t<name>' + object_name + '</name>\n')
            f.write('\t\t<pose>Unspecified</pose>\n')
            f.write('\t\t<truncated>0</truncated>\n')
            f.write('\t\t<difficult>0</difficult>\n')
            f.write('\t\t<bndbox>\n')
            f.write('\t\t\t<xmin>' + x_min + '</xmin>\n')
            f.write('\t\t\t<ymin>' + y_min + '</ymin>\n')
            f.write('\t\t\t<xmax>' + x_max + '</xmax>\n')
            f.write('\t\t\t<ymax>' + y_max + '</ymax>\n')
            f.write('\t\t</bndbox>\n')
            f.write('\t</object>\n')

        # Close the annotation tag once all the objects have been written to the file
        f.write('</annotation>\n')
    return True


def create_voc_data(
    root, 
    image_dir, 
    target_dir, 
    image_suffix, 
    target_suffix, 
    cls_dict, 
    data_ratio, 
    seed=10
): 
    # create directories
    root = Path(root)
    if os.path.isdir(root / 'VOCdevkit'):
        while True: 
            answer = input('VOCdevkit folder exists. Do you want to remove it?[Y/N]')  
            if answer.lower() == 'y': 
                shutil.rmtree(root)
                break

            elif answer.lower() == 'n': 
                return

    os.mkdir(root / 'VOCdevkit/')
    os.mkdir(root / 'VOCdevkit/VOC2007/')
    os.mkdir(root / 'VOCdevkit/VOC2007/Annotations/')
    os.mkdir(root / 'VOCdevkit/VOC2007/JPEGImages/')
    os.mkdir(root / 'VOCdevkit/VOC2007/ImageSets/')
    os.mkdir(root / 'VOCdevkit/VOC2007/ImageSets/Main/')
    
    image_names = [] 
    
    # cp images & xml
    for image_file in tqdm([file for file in os.listdir(image_dir) if file.endswith(image_suffix)]): 
        if target_suffix == '.txt': 
            res = txt2xml(
                os.path.join(target_dir, image_file.split('.')[0] + target_suffix), 
                os.path.join(root / 'VOCdevkit/VOC2007/Annotations/', image_file.split('.')[0] + '.xml'), 
                os.path.join(image_dir, image_file),
                cls_dict
            )        
        
        else: 
            res = os.path.isfile(os.path.join(target_dir, image_file.split('.')[0] + target_suffix))
 
        if res:
            shutil.copyfile(
                os.path.join(image_dir, image_file), 
                os.path.join(root/ 'VOCdevkit/VOC2007/JPEGImages/', image_file)
            )
            
        image_names.append(image_file.split('.')[0])
            
    # train test split
    image_names.sort()
    random.seed(seed) 
    random.shuffle(image_names)

    train_ratio, val_ratio = data_ratio 
    train_size, val_size = int(train_ratio * len(image_names)), int(val_ratio * len(image_names)) 

    train_image_names = image_names[:train_size]
    test_image_names = image_names[train_size:train_size + val_size]
    print(f'Training set has {len(train_image_names)} of images!')
    print(f'Training set has {len(test_image_names)} of images!')
    

    with open(root / 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'r') as f: 
        f.write('\n'.join(train_image_names))  

    with open(root / 'VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'r') as f: 
        f.write('\n'.join(test_image_names))  


if __name__ == '__main__': 
    create_voc_data(
        '/content/YOLOX', 
        '/content/train', 
        '/content/train', 
        '.png', '.txt', 
        {'0': 'car', '1': 'hov', '2': 'person', '3':'motorcycle'}, 
        data_ratio=(0.9, 0.1), 
        seed = 10
    )