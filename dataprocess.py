import numpy as np
import cv2
import os
import pandas as pd
import h5py
import glob
import json
import sys


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    #print(name)
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])
def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    #print(attrs)
    return attrs

def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file,'r') 
    all_rows = []
    print('image bounding box data construction starting...')
    bbox_df = pd.DataFrame([],columns=['height','img_name','label','left','top','width'])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df,pd.DataFrame.from_dict(row_dict,orient = 'columns')])
    bbox_df['bottom'] = bbox_df['top']+bbox_df['height']
    bbox_df['right'] = bbox_df['left']+bbox_df['width']
    print('finished image bounding box data construction...')
    return bbox_df

def write_txt(df):
    for img_name in df['img_name']:
        # print(img_name)
        file_name = img_name.split(".")[0]
        fp = open("./data/train/"+file_name+'.txt', "w")
        img_df = df[df['img_name'] == img_name]
        # class x_c, y_c, w, h
        for index, row in img_df.iterrows():
            label = row['label']
            if label == 10:
                label = 0
            img = cv2.imread('./data/train/'+img_name)

            img_h = img.shape[0]
            img_w = img.shape[1]

            x_c = (row['left']+row['right']) / 2 / img_w
            y_c = (row['top']+row['bottom']) / 2 / img_h
            w = row['width'] / img_w
            h = row['height'] / img_h
            fp.write(str(int(label))+" "+str(x_c)+" "+str(y_c)+" "+str(w)+" "+str(h)+'\n')
        fp.close()
    # write h w  
    fp.close()

def write_train_txt(df):
    fp = open("my/train.txt", 'w')
    for img_name in df['img_name']:
        fp.write("data/train/"+img_name+'\n')
    fp.close()
def write_test_txt(path):
    fp = open("my/test.txt", 'w')
    for filename in sorted(os.listdir(path), key=lambda x: int(x.split(".")[0])) :
        fp.write("data/test/"+filename+'\n')
    fp.close

    fp = open("img_wh.txt", "w")
    for filename in sorted(os.listdir(path), key=lambda x: int(x.split(".")[0])) :
        img = cv2.imread('./data/test/'+filename)
        h = img.shape[0]
        w = img.shape[1]
        # print(w, h)
        fp.write(filename+" "+str(w)+" "+str(h)+'\n')

def write_json(json_path, img_wh_path):
    f = open(json_path)
    data = json.load(f)
    f.close()

    wd_dic = {}
    f_wh = open(img_wh_path)
    for line in f_wh:
        wd_dic[line.split()[0]] = line.split()[1:]
    print(wd_dic)

    output = []
    for frame in data:
        file_name = frame['filename'].split('/')[2]
        # print(dic[file_name][0])
        frame_dict = {}
        bbox_ls = []
        score_ls = []
        label_ls = []
        for obj in frame['objects']:
            label = (obj['class_id'])
            if label == 0:
                label = 10
                print(label)
            score = (obj['confidence'])
            cx = obj['relative_coordinates']['center_x']
            cy = obj['relative_coordinates']['center_y']
            w = obj['relative_coordinates']['width']
            h = obj['relative_coordinates']['height']
            left = round((cx - w/2) * int(wd_dic[file_name][0]))
            right = round((cx + w/2) * int(wd_dic[file_name][0]))
            top = round((cy - h/2) * int(wd_dic[file_name][1]))
            bottom = round((cy + h/2) * int(wd_dic[file_name][1]))

            tup = (top, left, bottom, right)
            # print(file_name, wd_dic[file_name][0], wd_dic[file_name][1])
            # print(tup)
            bbox_ls.append(tup)
            label_ls.append(label)
            score_ls.append(score)

        frame_dict["bbox"] = bbox_ls
        frame_dict["score"] = score_ls
        frame_dict["label"] = label_ls
        output.append(frame_dict)
    # print(output)
    with open('submission.json', "w") as fjson:
        json.dump(output, fjson)

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        status = 'preprocess'
    elif sys.argv[1] == 'postprocess':
        status = 'postprocess'
    else:
        status = 'preprocess'
    if status == 'preprocess':
        print("preprocessing...")
        bbox_df = img_boundingbox_data_constructor("./data/train/digitStruct.mat")
        print(bbox_df)
        bbox_df.to_pickle("bbox_df.pkl")
        # bbox_df = pd.read_pickle("bbox_df.pkl")
        print(bbox_df)
        write_txt(bbox_df)
        write_train_txt(bbox_df)
        write_test_txt('data/test')
    elif status == 'postprocess':
        print("postprocessing...")
        write_json('result.json', 'img_wh.txt')
