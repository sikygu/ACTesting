import json
import torch
from flair.data import Sentence
from flair.nn import Classifier
import stanza
import tqdm
import random

def readfile(textpath):
    with open(textpath, 'r') as f:
        myList = json.load(f)
    return myList

def writefile(listdata, filepath):
    with open(filepath, 'w') as f:
        json.dump(listdata, f)
    return 0

def getcocoword():
    '''get 80 classes word'''
    with open('/home/t2i/cocodata/annotations_trainval2014/instances_train2014.json', 'r') as f:
        data = json.load(f)
    categories = data['categories']
    category_names = [category['name'] for category in categories]
    return category_names

def stanza_relation():
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', download_method=None)
    return nlp

def keywordcoco(oldsen, cocowords):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(oldsen)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    keyword=0
    for word in filtered_sentence:
        if word in cocowords:
            keyword = word
    return keyword

def getsamere(res):
    nlp = stanza_relation()
    file = readfile(textpath)
    cocowords = getcocoword()
    relationdict = {}
    for item in res:
        relationdict[item] = []

    for item in file:
        id = item['image_id']
        line = item['oldcaption']
        relations = nlp(line)
        for sent in relations.sentences:
            for word in sent.words:
                if word.text in cocowords:
                    re = word.deprel
                    w = word.text
                    # print(relationdict[re])
                    relationdict[re].append([w, line, id])
                    # relationdict[re].append(w)
        print(id)
    writefile(relationdict, '/home/MT/textdata/word_relation.json')
    return relationdict

def getnsubj(file): 
    data = readfile(file)
    data_nsubj = data['nsubj']+data['nsubj:pass']
    nlp = stanza_relation()
    result = []
    count = 0
    for item in data_nsubj:
        cocoword = item[0]
        sen = item[1]
        relations = nlp(sen)
        find = False
        for word in relations.sentences[0].words:
            if word.head == cocoword and deprel =='compound':
                find = True 
                break
        if find!=True:
            result.append(item)
        count = count + 1
        print(count)
    writefile(result, '/home/MT/textdata/word_relation_nsubj.json')

def changed(file, output):
    data = readfile(file)
    words = []
    sum = len(data)
    sortlist = []
    for i in range(sum):
        sortlist.append(i) # 0.1.2.3
    random.shuffle(sortlist)
    for i in range(len(sortlist)):
        while i == sortlist[i] - 1:
            random.shuffle(sortlist) # 3.2.0.1

    new_data = []
    for item in range(len(data)):
        newdata = {}
        image_id = data[item][2]
        oldword = data[item][0]
        oldsen = data[item][1]

        changeword = data[sortlist[item]][0]
        changesen = oldsen.replace(oldword, changeword)

        newdata['image_id'] = image_id
        newdata['oldcaption'] = oldsen
        newdata['oldword'] = oldword
        newdata['newcaption'] = changesen
        newdata['newword'] = changeword

        new_data.append(newdata)

    writefile(new_data, output)

def getobj(file): 
    data = readfile(file)
    data_obj = data['obj']
    writefile(data_obj, '/home/MT/textdata/word_relation_obj.json')

if __name__ == "__main__":
    res = ['compound', 'nmod', 'root', 'conj', 'obl', 'nsubj', 'acl:relcl', 'obj', 'nsubj:pass', 'advcl', 'amod', 'obl:agent', 'obl:tmod', 'appos', 'acl', 'nmod:poss', 'dislocated', 'parataxis', 'xcomp', 'iobj', 'nsubj:outer', 'obl:npmod', 'ccomp', 'nmod:npmod', 'advmod']
    textpath = '/home/MT/textdata/coco2014valimage-caption.json'
    print('1')
    # getsamere(res) # word_relation.json
    print('2')
    # getnsubj('/home/MT/textdata/word_relation.json') 
    print('3')
    getobj('/home/MT/textdata/word_relation.json') 

    # changed('/home/MT/textdata/word_relation_nsubj.json', '/home/MT/textdata/word_relation_nsubj_re.json')
    changed('/home/MT/textdata/word_relation_obj.json', '/home/MT/textdata/word_relation_obj_re.json')
