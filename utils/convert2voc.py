import os
import shutil
import random 
import cv2
from .config import cls_dict


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
            print(f'{txt_path} has no label!')
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


def create_VOCdevkit(image_dir, target_dir, image_suffix, target_suffix, cls_dict): 
    # create directories
    os.mkdir('./VOCdevkit/')
    os.mkdir('./VOCdevkit/VOC2007/')
    os.mkdir('./VOCdevkit/VOC2007/Annotations/')
    os.mkdir('./VOCdevkit/VOC2007/JPEGImages/')
    os.mkdir('./VOCdevkit/VOC2007/ImageSets/')
    os.mkdir('./VOCdevkit/VOC2007/ImageSets/Main/')
    
    image_names = [] 
    
    # cp images & xml
    for image_file in [file for file in os.listdir(image_dir) if file.endswith(image_suffix)]: 
        if target_suffix == '.txt': 
            res = txt2xml(
                os.path.join(target_dir, image_file.split('.')[0] + target_suffix), 
                os.path.join('./VOCdevkit/VOC2007/Annotations/', image_file.split('.')[0] + '.xml'), 
                os.path.join(image_dir, image_file),
                cls_dict
            )        
        
        else: 
            res = os.path.isfile(os.path.join(target_dir, image_file.split('.')[0] + target_suffix))
 
        if res:
            shutil.copyfile(
                os.path.join(image_dir, image_file), 
                os.path.join('./VOCdevkit/VOC2007/JPEGImages/', image_file)
            )
            
        image_names.append(image_file.split('.')[0])
            
    with open('./VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w') as f:
        f.write('\n'.join(image_names))


def train_test_split(dir_name='./VOCdevkit/VOC2007/ImageSets/Main/', test_ratio=0.1, seed=10): 
    assert 'trainval.txt' in os.listdir(dir_name)
    random.seed(seed)
    with open(os.path.join(dir_name, 'trainval.txt'), 'r') as f: 
        total_names = [line.strip() for line in f.readlines() if line.strip()] 
        random.shuffle(total_names)
        test_size = int(len(total_names) * test_ratio) 
        train_names = total_names[test_size:]
        test_names = total_names[:test_size]
        print(f'Training set has {len(train_names)} of images!')
        print(f'Testing set has {len(test_names)} of images!')

    with open(os.path.join(dir_name, 'trainval.txt'), 'w') as f:
        f.write('\n'.join(train_names))   

    with open(os.path.join(dir_name, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_names))


if __name__ == '__main__': 
    create_VOCdevkit('/content/train', '/content/train', '.png', '.txt', cls_dict)
    train_test_split('/content/YOLOX/VOCdevkit/VOC2007/ImageSets/Main/')