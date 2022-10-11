import os
import glob
import cv2
from config import cls_dict


def txt2xml(txt_file, xml_file, image_path, image_dir):
    image_height, image_width, _ = cv2.imread(image_path).shape
    image_name = os.path.basename(image_path)
    image_path = os.path.join(image_dir, image_name)
    
    with open(txt_file, 'r') as f: 
        s = f.read().strip()
        if s: 
            lines = s.split('\n')
        else: 
            print(txt_file)
            lines = []

    with open(xml_file, 'w') as f:
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
        f.close() # Close the file

def generate_xml_files(txt_paths, xml_dir, image_dir): 
    for txt_path in txt_paths:
        xml_path = os.path.join(xml_dir, os.path.basename(txt_path).replace('.txt', '.xml')) 
        image_path = os.path.join(image_dir, os.path.basename(txt_path).replace('.txt', '.png'))
        txt2xml(txt_path, xml_path, image_path, image_dir)


if __name__ == '__main__': 
    txt_paths = glob.glob('/content/train/*.txt')
    generate_xml_files(txt_paths, '/content/xml_annotation/', '/content/train/')