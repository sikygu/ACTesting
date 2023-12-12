import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from keybert import KeyBERT
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
from transformers import BertTokenizer, BertModel, BertForMaskedLM, pipeline
import torch
import time
from tqdm import tqdm
from torch.utils.data import Dataset
import time
import re

modelpath = '/home/EDA/pretrain/checkpoint-4700/'
textpath = '/home/MT/textdata/coco2014valimage-caption.json'


outputpath = '/home/MT/textdata/0324MR.json'
maskfilepath = '/home/MT/textdata/0324mask.json'

def getcocoword():
    '''get 80 classes word'''
    with open('/home/t2i/cocodata/annotations_trainval2017/instances_val2017.json', 'r') as f:
        data = json.load(f)
    categories = data['categories']
    category_names = [category['name'] for category in categories]
    # return random.choice(category_names)
    return category_names

# 暂时没用
def addword_exit():
    # get posword
    prepositions = ['at', 'atop', 'onto', 'opposite to', 'out', 'round', 'past', 'around', 'off',
     'from', 'in', 'down', 'across', 'of', 'with', 'in front of', 'near', 'underneath',
     'but', 'upon', 'before', 'below', 'toward', 'away from', 'inside', 'under', 'far from',
     'to', 'beneath', 'behind', 'towards', 'beside', 'since', 'for', 'without',
     'out of', 'beyond', 'outside of', 'up', 'into', 'between', 'unto', 'on', 'next to',
     'above', 'within', 'along', 'among', 'over', 'by']

    return random.choice(prepositions)

def keynnword(oldsen):
    # get key word nn
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(oldsen)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    tagged = nltk.pos_tag(filtered_sentence)
    keywords = [word for word, pos in tagged if pos == 'NN' or pos == 'NNS']
    if keywords == []:
        return 0
    else:
        return keywords[0]

def keywordcoco(oldsen, cocowords):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(oldsen)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    keyword=0
    for word in filtered_sentence:
        if word in cocowords:
            keyword = word
    return keyword

def generate_new_sen_randompos(oldsen, cocoword, keywords, pos):
    # random_num = random.choice([-1, 1]) 
    for i in range(len(words)):
        if words[i] == keywords:
            words.insert(i + 1, pos)
            words.insert(i + 2, cocoword)
            break
    sen_without_pos = " ".join(words)
    return sen_without_pos


def Single_generate_new_sen_bert(oldsen):
    cocoword = getcocoword()
    keywords = keynnword(oldsen)
    words = oldsen.split()
    for i in range(len(words)):
        if words[i] == keywords:
            words.insert(i + 1, '[MASK]')
            words.insert(i + 2, cocoword)
            break

    sen_without_pos = " ".join(words)

    model = BertForMaskedLM.from_pretrained('/home/EDA/pretrain/checkpoint-4700/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    sen_with_pos = nlp(sen_without_pos)
    return sen_with_pos[0]['sequence']


def readfile(textpath):
    with open(textpath, 'r') as f:
        myList = json.load(f)
    return myList

def writefile(listdata, filepath):
    with open(filepath, 'w') as f:
        json.dump(listdata, f)
    return 0

def getdata(jsonlist, maskfile):
    jsonlist = readfile(textpath)
    cocowords = getcocoword() 
    for item in jsonlist:
        line = item['oldcaption']
        cocoword = random.choice(cocowords) 
        while cocoword in line: 
            cocoword = random.choice(cocowords)
        item['cocoword'] = cocoword
        keywords = keynnword(line)
        keywords = keywordcoco(line, cocowords)
        if keywords == 0: 
            item['maskcaption'] = 0
            print(item['id'], line)
            continue
        words = re.findall(r'\b\w+\b', line)
        for i in range(len(words)): 
            if words[i] == keywords:
                words.insert(i + 1, '[MASK]')
                words.insert(i + 2, cocoword)
                break
        sen_without_pos = " ".join(words)
        item['maskcaption'] = sen_without_pos

    writefile(jsonlist, maskfile)
    return 0

class MyDataset(Dataset):
    def __init__(self, maskfilepath):
        super().__init__()
        # self.device_index = device_index
        self.path = maskfilepath
        self.data = self.getdata()
        self.dict = self.getorigion()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def getdata(self):
        
        maskcap = []
        maskjsonlist = readfile(self.path)
        for i in range(len(maskjsonlist)):
            if maskjsonlist[i]['maskcaption'] == 0:
                continue
            elif not '[MASK]' in maskjsonlist[i]['maskcaption']:
                continue
            else:
                maskcap.append(maskjsonlist[i]['maskcaption'])
        return maskcap

    def getorigion(self):
        maskjsonlist = readfile(self.path)
        return maskjsonlist

def Batch_usebert():
    
    model = BertForMaskedLM.from_pretrained(modelpath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    pipe = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device='cuda:0',
        framework="pt",
        batch_size=16
    )
    return pipe

def run():
    outputl = []
    pipe = Batch_usebert()
    data = MyDataset(maskfilepath)
    origion = data.getorigion() 
    mask_cap = {}
    for out in tqdm(pipe(data)):
        pred = out[0]['sequence']
        mask_cap[data[i]] = pred
        i = i + 1

    for item in origion:
        item['newcaption'] = 0 

    for item in origion:
        for k, v in mask_cap.items():
            if item['maskcaption'] == k:
                item['newcaption'] = v

    writefile(origion, outputpath)
    return origion

if __name__ == "__main__":
    print('generate mask text...')
    start_time = time.time()
    getdata(textpath, maskfilepath)
    print('generate bert word')
    run()
    end_time = time.time()
    print("getdata took", end_time - start_time, "seconds to run")


