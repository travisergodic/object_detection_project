import os
import cv2
from config import color_dict

def demo_result(img_path, txt_file, save_dir): 
    img = cv2.imread(img_path)
    with open(txt_file, 'r') as f: 
        s = f.read().strip()
        if s: 
            lines = s.split('\n')
        else: 
            lines = []

    for line in lines: 
        cls, x, y, w, h = line.split(',')
        x_min, y_min = int(x), int(y)
        x_max, y_max = int(x) + int(w), int(y) + int(h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_dict[cls], 2)

    cv2.imwrite(os.path.join(save_dir, 'demo_' + os.path.basename(img_path)))