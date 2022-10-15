import os
import pandas as pd 
from statistics import mean
from collections import defaultdict

def bboxes_statistics(dir_name, target_suffix):
    obj_num_dict = dict()
    obj_info_list = list()
    cls_num_dict = defaultdict(int) 
    for filename in os.listdir(dir_name): 
        if not filename.endswith(target_suffix): 
            continue

        with open(os.path.join(dir_name, filename)) as f: 
            img_info = f.readlines()

        # count object number 
        obj_num_dict[filename] = len(img_info) 

        # get object info 
        for obj_info in img_info: 
            c, x_min, y_min, x_max, y_max = [int(ele.strip()) for ele in obj_info.split(',')]
            obj_info_list.append(
                (int(x_min + x_max)/2, int(y_min + y_max)/2, x_max - x_min, y_max - y_min)
            )
            cls_num_dict[c] += 1

    print(f'Maximum number of objects: ( {(max(obj_num_dict, key=obj_num_dict.get))} , {max(obj_num_dict.values())} ) !')
    print(f'No object files: {[filename for filename in obj_num_dict if obj_num_dict[filename] == 0]}!')
    print(f'Average number of objects: {mean(obj_num_dict.values())}')
    print(f'Average object area: {mean([w * h for _, _, w, h in obj_info_list])}')
    print(f'Class frequency count: {cls_num_dict}')