import json
import random
import string

# Load annotations
with open('/home/t2i/cocodata/annotations_trainval2017/captions_val2017.json', 'r') as f:
    annotations = json.load(f)

with open('/home/t2i/cocodata/annotations_trainval2017/instances_val2017.json', 'r') as f:
    categories = json.load(f)

def getcocoword():
    '''get 80 classes word'''
    with open('/home/t2i/cocodata/annotations_trainval2014/instances_train2014.json', 'r') as f:
        data = json.load(f)
    categories = data['categories']
    category_names = [category['name'] for category in categories]
    # return random.choice(category_names)
    return category_names

def contains_word(sentence, word_list):
    words_found=[]
    for word in word_list:
        if word in sentence:
            words_found.append(word)
    if len(words_found)==0:
        return False
    else:
        return words_found

# choose random caption with image
captions = {}
for annot in annotations['annotations']:
    image_id = annot['image_id']
    caption = annot['caption']
    if image_id not in captions:
        captions[image_id] = []
    captions[image_id].append(caption)

captions_new = {}
captionword = {}
cococatgories = getcocoword()
for k, v in captions.items():
    # random_string = ""
    get = False
    for i in v:
        # get = contains_word(i, cococatgories)
        get = True
        if not get: 
            continue
        else:
            captions_new[k] = i
            captionword[k] = get
            break

def lowercase_except_first(s):
    return s[0] + s[1:].lower()
def removepoint(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

new = []
i = 1
for image_id, v in captionword.items():
    new_unit = {}
    imageid = image_id 
    imagecaption = captions_new[imageid]# 
    oldcaptionword = v 
    imagecaption = lowercase_except_first(imagecaption)
    # imagecaption = removepoint(imagecaption)
    imagecaption = imagecaption.split('\n')[0] 

    if imagecaption=='': 
        continue
    new_unit['id'] = i
    new_unit['image_id'] = imageid
    new_unit['oldcaption'] = imagecaption
    # new_unit['oldcaption_obj'] = oldcaptionword
    new.append(new_unit)
    i = i + 1

def write_list_to_file(lst, filename):
    with open(filename, 'w') as f:
        json.dump(new, f)

write_list_to_file(new, '../textdata/coco2017valimage-caption.json')