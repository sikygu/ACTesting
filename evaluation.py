import torch
import json
import h5py
import random
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import argparse
import os

path = '/ACTesting/test_data'

def check_elements_in_same_list(lists, a, b):
    for lst in lists:
        if a in lst and b in lst:
            return True
    return False

def get_image_index(image_id, data):
    # Extract image IDs from image paths
    image_ids = [os.path.basename(path).split('_')[0] for path in data['idx_to_files']]

    # Find the index of the given image ID
    if image_id in image_ids:
        return image_ids.index(image_id)
    else:
        return -1

def evaluation(json_path):
    json_name = json_path.split('/')[-1]+'.json'
    custom_prediction = json.load(open(os.path.join(json_path, 'custom_prediction.json')))
    custom_data_info = json.load(open(os.path.join(json_path, 'custom_data_info.json')))
    captions_file = json.load(open(os.path.join(path, json_name)))

    image_sum = len(custom_data_info['idx_to_files']) 
    entities_sum = 0
    rel_sum = 0
    for item in captions_file:
        img_id = item['image_id']
        entities = item['classes']
        rel = item['triples']
        entities_sum = entities_sum + len(entities)
        rel_sum = rel_sum + len(rel)
    # print(json_path)
    # print('sum of entities:', entities_sum)
    # print('sum of rel:', rel_sum)
    # print('sum of image:', image_sum)
    sum_error_e=0
    sum_error_r=0

    for img in range(image_sum):
        img_id = custom_data_info['idx_to_files'][img].split('/')[-1].split('_')[0]
        # print(img_id)
        for item in captions_file:
            if str(item['image_id']) == img_id:
                entities = item['classes']
                rel = item['triples']
                # break
        real_entities = entities
        real_rel = rel
        box_labels, rel_labels = get_bbox_re(custom_data_info,custom_prediction, img)
        
        error_e = 0
        error_r = 0
        for i in real_entities:
            if i not in box_labels:
                error_e = error_e + 1

        for i in real_rel:
            find = 0
            for j in rel_labels:
                if j[1]==i[1]:
                    find=1
                    break
            if find==0:
                error_r = error_r + 1


        sum_error_e = sum_error_e + error_e
        sum_error_r = sum_error_r + error_r


    print('error rate of entities: ', sum_error_e/entities_sum)
    print('error rate of relationships: ', sum_error_r/rel_sum)

    return sum_error_e/entities_sum, sum_error_r/rel_sum

def MR_EC(test_file, original_file):
    json_name = test_file.split('/')[-1]+'.json'
    origin_name = original_file.split('/')[-1]+'.json'
    custom_prediction = json.load(open(os.path.join(test_file, 'custom_prediction.json')))
    custom_data_info = json.load(open(os.path.join(test_file, 'custom_data_info.json')))
    captions_file = json.load(open(os.path.join(path, json_name)))

    origin_custom_prediction = json.load(open(os.path.join(original_file, 'custom_prediction.json')))
    origin_custom_data_info = json.load(open(os.path.join(original_file, 'custom_data_info.json')))
    origin_captions_file = json.load(open(os.path.join(path, origin_name)))

    image_sum = len(custom_data_info['idx_to_files']) 
    entities_sum = 0
    rel_sum = 0
    dataline = 10
    for item in captions_file:
        img_id = item['image_id']
        entities = item['classes']
        rel = item['triples']
        entities_sum = entities_sum + len(entities)
        rel_sum = rel_sum + len(rel)
    # print('sum of entities:', entities_sum)
    # print('sum of rel:', rel_sum)
    # print('sum of image:', image_sum)
    sum_error_e=0
    sum_error_r=0

    for img in range(image_sum):
        img_id = custom_data_info['idx_to_files'][img].split('/')[-1].split('_')[0]
        # print(img_id)
        for item in captions_file:
            if str(item['image_id']) == img_id:
                EC_entities = item['classes']
                EC_rel = item['triples']
                replace = item['EC']
                break
        
        for item in origin_captions_file:
            if str(item['image_id']) == img_id:
                O_entities = item['classes']
                O_rel = item['triples']
                break
        
        box_labels, rel_labels = get_bbox_re(custom_data_info, custom_prediction, img)
        img_index_origin = get_image_index(img_id, origin_custom_data_info)
        # print(img_index_origin)
        origin_box_labels, origin_rel_labels = get_bbox_re(origin_custom_data_info, origin_custom_prediction, img_index_origin)
        
        origin_rel_labels = origin_rel_labels[:dataline]
        rel_labels = rel_labels[:dataline]

        error_e = 0
        error_r = 0

        if not replace[1] in box_labels:
            sum_error_e = sum_error_e + 1
        # elif not replace[0] in origin_box_labels:
        #     error_e = error_e + 1

        for i in origin_rel_labels:
            find = 0
            for j in rel_labels:
                if j[1] == i[1]:
                    find=1
                    break
            
            if find==0:
                sum_error_r = sum_error_r + 1
                break
            else:
                continue
            
    return sum_error_e/image_sum, sum_error_r/image_sum

def MR_ER_R(test_file, original_file):
    json_name = test_file.split('/')[-1]+'.json'
    # origin_name = original_file.split('/')[-1]+'.json'
    custom_prediction = json.load(open(os.path.join(test_file, 'custom_prediction.json')))
    custom_data_info = json.load(open(os.path.join(test_file, 'custom_data_info.json')))
    captions_file = json.load(open(os.path.join(path, json_name)))

    origin_custom_prediction = json.load(open(os.path.join(original_file, 'custom_prediction.json')))
    origin_custom_data_info = json.load(open(os.path.join(original_file, 'custom_data_info.json')))
    # origin_captions_file = json.load(open(os.path.join(path, origin_name)))

    image_sum = len(custom_data_info['idx_to_files']) 
    entities_sum = 0
    rel_sum = 0
    dataline = 20
    for item in captions_file:
        img_id = item['image_id']
        entities = item['classes']
        rel = item['triples']
        entities_sum = entities_sum + len(entities)
        rel_sum = rel_sum + len(rel)
    # print('sum of entities:', entities_sum)
    # print('sum of rel:', rel_sum)
    # print('sum of image:', image_sum)
    sum_error_e=0
    sum_error_r=0

    for img in range(image_sum):
        img_id = custom_data_info['idx_to_files'][img].split('/')[-1].split('_')[0]
        # print(img_id)
        for item in captions_file:
            if str(item['image_id']) == img_id:
                bbox = item['classes']
                re_class = item['re_class']
                re_triple = item['re_triple']
                break
        
        box_labels, rel_labels = get_bbox_re(custom_data_info, custom_prediction, img)
        img_index_origin = get_image_index(img_id, origin_custom_data_info)
        origin_box_labels, origin_rel_labels = get_bbox_re(origin_custom_data_info, origin_custom_prediction, img_index_origin)
        
        origin_rel_labels = origin_rel_labels[:dataline]
        rel_labels = rel_labels[:dataline]

        error_e = 0
        error_r = 0

        if re_class in box_labels:
            sum_error_e = sum_error_e + 1
            

        else:
            for i in bbox:
                if i not in box_labels:
                    sum_error_e = sum_error_e + 1
                    break

        
        find = 0 
        for i in re_triple:
            if i not in rel_labels:
                # error_r = error_r + 1
                find = 1
                break
        
        if find==0:
            sum_error_r = sum_error_r + 1
            continue

        else:
            # find=0
            for i in rel_labels:
                find = 0
                for j in origin_rel_labels:
                    if j[1] == i[1]:
                        find=1
                        break
                if find==0:
                    sum_error_r = sum_error_r + 1
                    break
        

    # print('error rate of entities: ', sum_error_e/image_sum)
    # print('error rate of relationships: ', sum_error_r/image_sum)
    
    return sum_error_e/image_sum, sum_error_r/image_sum

def MR_ER_A(test_file, original_file):
    json_name = test_file.split('/')[-1]+'.json'
    # origin_name = original_file.split('/')[-1]+'.json'
    custom_prediction = json.load(open(os.path.join(test_file, 'custom_prediction.json')))
    custom_data_info = json.load(open(os.path.join(test_file, 'custom_data_info.json')))
    captions_file = json.load(open(os.path.join(path, json_name)))

    origin_custom_prediction = json.load(open(os.path.join(original_file, 'custom_prediction.json')))
    origin_custom_data_info = json.load(open(os.path.join(original_file, 'custom_data_info.json')))
    # origin_captions_file = json.load(open(os.path.join(path, origin_name)))

    image_sum = len(custom_data_info['idx_to_files']) # 
    entities_sum = 0
    rel_sum = 0
    dataline = 20
    for item in captions_file:
        img_id = item['image_id']
        entities = item['classes']
        rel = item['triples']
        entities_sum = entities_sum + len(entities)
        rel_sum = rel_sum + len(rel)
    # print('sum of entities:', entities_sum)
    # print('sum of rel:', rel_sum)
    # print('sum of image:', image_sum)
    sum_error_e=0
    sum_error_r=0

    for img in range(image_sum):
        img_id = custom_data_info['idx_to_files'][img].split('/')[-1].split('_')[0]
        # print(img_id)
        for item in captions_file:
            if str(item['image_id']) == img_id:
                new_class = item['new_class']
                new_re = item['new_re']
                break
        
        box_labels, rel_labels = get_bbox_re(custom_data_info, custom_prediction, img)
        img_index_origin = get_image_index(img_id, origin_custom_data_info)
        origin_box_labels, origin_rel_labels = get_bbox_re(origin_custom_data_info, origin_custom_prediction, img_index_origin)
        origin_rel_labels = origin_rel_labels[:dataline]
        rel_labels = rel_labels[:dataline]

        # error_e = 0
        # error_r = 0

        if new_class not in box_labels:
            # error_e = error_e + 1
            sum_error_e = sum_error_e + 1

        find = 0
        for i in rel_labels:
            if new_re == i[1]:
                find = 1
                break

        if find == 0:
            sum_error_r = sum_error_r + 1
            continue
        
        else:
            for i in origin_rel_labels:
                find = 0
                for j in rel_labels:
                    if j[1] == i[1]:
                        find = 1
                        break
                if find == 0:
                    sum_error_r = sum_error_r + 1
                    break
    
    return sum_error_e/image_sum, sum_error_r/image_sum

def MR_letter(test_file, original_file):
    json_name = test_file.split('/')[-1]+'.json'
    origin_name = original_file.split('/')[-1]+'.json'
    custom_prediction = json.load(open(os.path.join(test_file, 'custom_prediction.json')))
    custom_data_info = json.load(open(os.path.join(test_file, 'custom_data_info.json')))
    captions_file = json.load(open(os.path.join(path, json_name)))

    origin_custom_prediction = json.load(open(os.path.join(original_file, 'custom_prediction.json')))
    origin_custom_data_info = json.load(open(os.path.join(original_file, 'custom_data_info.json')))
    origin_captions_file = json.load(open(os.path.join(path, origin_name)))

    image_sum = len(custom_data_info['idx_to_files']) # 
    entities_sum = 0
    rel_sum = 0
    dataline = 20
    for item in captions_file:
        img_id = item['image_id']
        entities = item['classes']
        rel = item['triples']
        entities_sum = entities_sum + len(entities)
        rel_sum = rel_sum + len(rel)
    # print('sum of entities:', entities_sum)
    # print('sum of rel:', rel_sum)
    # print('sum of image:', image_sum)
    sum_error_e=0
    sum_error_r=0

    for img in range(image_sum):
        img_id = custom_data_info['idx_to_files'][img].split('/')[-1].split('_')[0]
        # print(img_id)
        for item in captions_file:
            if str(item['image_id']) == img_id:
                boxx = item['classes']
                entities = item['classes']
                rel = item['triples']
                break
        
        box_labels, rel_labels = get_bbox_re(custom_data_info, custom_prediction, img)
        img_index_origin = get_image_index(img_id, origin_custom_data_info)
        # print(img_index_origin)
        origin_box_labels, origin_rel_labels = get_bbox_re(origin_custom_data_info, origin_custom_prediction, img_index_origin)
        
        origin_rel_labels = origin_rel_labels[:dataline]
        rel_labels = rel_labels[:dataline]

        error_e = 0
        error_r = 0

        
        for i in boxx:
            if i not in origin_box_labels:
                error_e = error_e + 1
                break 

        for i in rel_labels:
            find = 0
            for j in origin_rel_labels:
                if j[1] == i[1]:
                    find=1
                    break
            if find==0:
                error_r = error_r + 1
                break
            else:
                continue
        
        sum_error_e = sum_error_e + error_e
        sum_error_r = sum_error_r + error_r

    
    return sum_error_e/image_sum, sum_error_r/image_sum

def get_bbox_re(custom_data_info,custom_prediction,image_idx):
    box_topk = 20 # select top k bounding boxes
    rel_topk = 20 # select top k relationships
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']
    image_path = custom_data_info['idx_to_files'][image_idx]
    boxes = custom_prediction[str(image_idx)]['bbox'][:box_topk]
    box_labels = custom_prediction[str(image_idx)]['bbox_labels'][:box_topk]
    box_scores = custom_prediction[str(image_idx)]['bbox_scores'][:box_topk]
    all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
    all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
    all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']
    for i in range(len(box_labels)):
        # print(box_labels[i])
        box_labels[i] = ind_to_classes[box_labels[i]]
    rel_labels = []
    rel_scores = []

# box_labels 0: tail; score: 0.7857648730278015
# rel_labels 0: 20_cat => has => 0_tail; score: 0.9304343461990356
    for i in range(len(all_rel_pairs)):
        if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
            rel_scores.append(all_rel_scores[i])
            rel_entity = set(rel_scores)
            # triple = [box_labels[all_rel_pairs[i][0]], ind_to_predicates[all_rel_labels[i]], box_labels[all_rel_pairs[i][1]]]
            triple=[]
            triple.append(box_labels[all_rel_pairs[i][0]])
            triple.append(ind_to_predicates[all_rel_labels[i]])
            triple.append(box_labels[all_rel_pairs[i][1]])

            rel_labels.append(triple)

    rel_labels = rel_labels
    return box_labels, rel_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    print("Caculate ...")

    # ablation study
    ablationfiles = ['new-captions_val2017_new', 'ER-R_captions_val2017', 'ER-A_captions_val2017', 'EC_captions_val2017', 'EC+ER-R_captions_val2017', 'EC+ER-A_captions_val2017']
    ablationresults = {}
    model_name = args.model_path.split('/')[-1]
    if model_name == '':
        model_name = args.model_path.split('/')[-2]

    for item in ablationfiles:
        print(item)
        inputpath = os.path.join(args.model_path, item) #
        error_rate_en, error_rate_re = evaluation(inputpath)
        ablationresults[item] = [error_rate_en, error_rate_re]
    
    file_name1 = os.path.join('/ACTesting/results/ablation_'+ model_name + '.txt')
    with open(file_name1, "w") as f:
        for k,v in ablationresults.items():
            f.write("{}: ablation: error_rate_en {:.4f}, error_rate_re {:.4f}\n".format(k, v[0], v[1]))

    # MRs
    MRfiles = ['letter_captions_val2017','ER-R_captions_val2017','EC_captions_val2017','ER-A_captions_val2017']
    MRresults = {}
    for item in MRfiles:
        print(item)
        test_file = os.path.join(args.model_path, item)
        original_file = os.path.join(args.model_path, 'new-captions_val2017_new')
        if item == 'EC_captions_val2017':
            error_rate_en, error_rate_re = MR_EC(test_file, original_file)
            MRresults[item] = error_rate_en, error_rate_re
        elif item == 'ER-R_captions_val2017':
            error_rate_en, error_rate_re = MR_ER_R(test_file, original_file)
            MRresults[item] = error_rate_en, error_rate_re
        elif item == 'ER-A_captions_val2017':
            error_rate_en, error_rate_re = MR_ER_A(test_file, original_file)
            MRresults[item] = error_rate_en, error_rate_re
        else:
            error_rate_en, error_rate_re = MR_letter(test_file, original_file)
            MRresults[item] = error_rate_en, error_rate_re
        print('error rate of entities: ', error_rate_en)
        print('error rate of relationships: ', error_rate_re)

    file_name2 = os.path.join('/ACTesting/results/MRs_'+ model_name + '.txt')
    with open(file_name2, "w") as f:
        for k,v in MRresults.items():
            f.write("{} MRs: error_rate_en: {:.4f} error_rate_re: {:.4f}\n".format(k, v[0], v[1]))



    