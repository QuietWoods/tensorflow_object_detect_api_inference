import os
import sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    count_mark = 10000000
    for xml_file in glob.glob(path + '/*.xml'):
        print('-------------begin---per xml file------------')
        print('xml_file', xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        count_mark -= 1
        if count_mark < 0:
            break
        print(os.path.basename(xml_file).replace('xml', 'jpg'))
        for member in root.findall('object'):
            name = member.find('name')
            print('class: ', name.text)
            bbox = member.find('bndbox')
            print('xmin', bbox[0].text, end=",")
            print('ymin', bbox[1].text, end=",")
            print('xmax', bbox[2].text, end=",")
            print('ymax', bbox[3].text)
            value = (os.path.basename(xml_file).replace('xml', 'jpg'),
                     int(root.find('size')[0].text),  # width
                     int(root.find('size')[1].text),  # height
                     name.text,
                     int(float(bbox[0].text)),
                     int(float(bbox[1].text)),
                     int(float(bbox[2].text)),
                     int(float(bbox[3].text))
                     )
            xml_list.append(value)
        print('-----------------end---per xml file-----------')
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print('ok')
    return xml_df


def convert_xml2csv(xml_path_dir, test_data_info_path):
    
    XML_PATH_DIR = xml_path_dir
    # CSV_PATH = "test_labels.csv"
    CSV_PATH = test_data_info_path


    xml_dirs = os.listdir(XML_PATH_DIR)
    print('----------before sort xml dirs--------------')
    print(xml_dirs)
    xml_dirs.sort()
    print('----------after sort xml dirs---------------')
    print(xml_dirs)
    is_first_xml = True
    for directory in xml_dirs:
        if not directory.endswith('Annotations'):
            continue
        xml_path = os.path.join(XML_PATH_DIR, directory)
        xml = glob.glob(xml_path + '/*.xml')
        print('xml',xml_path)
        # image_path = os.path.join(os.getcwd(), 'merged_xml')
        xml_df = xml_to_csv(xml_path)
        # xml_df.to_csv('whsyxt.csv', index=None)
        if is_first_xml:
            xml_df.to_csv(CSV_PATH,mode='a', index=None)
            is_first_xml = False
        else:
            xml_df.to_csv(CSV_PATH,mode='a', header=False, index=None)
        print('Successfully converted xml to csv.')
#
#
# if __name__ == '__main__':
#     assert len(sys.argv)==2, 'python xml_path_dir'
#     xml_path_dir = sys.argv[1]
#     if not os.path.exists(xml_path_dir):
#         print(xml_path_dir)
#     convert_xml2csv(xml_path_dir, test_data_info_path='test_labels.csv')

