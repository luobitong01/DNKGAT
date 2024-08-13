from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os
import torch
import re
import string
import torch
import torch.nn.functional as F
from path_zh import *
import csv
from config import *
import numpy as np
from data import *
import jieba
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import BertTokenizer

import warnings
warnings.filterwarnings('ignore')

class StringProcess(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr,' ',string)
        string = re.sub(r4,' ',string)
        string = string.strip().lower()
        string = self.remove_stopword(string)
        return string

    def clean_str_zh(self, string):
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        string = re.sub(r4, ' ', string)
        string = string.strip()
        string = self.remove_stopword_zh(string)
        return string

    def clean_str_BERT(self,string):
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
        r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        string = re.sub(r4, ' ', string)
        return string

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)
        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def remove_stopword_zh(self, string):
        stopwords = []
        with open('../data/weibo/stop_words.txt', 'r', encoding='utf-8')as f:
            txt = f.readlines()
        for line in txt:
            stopwords.append(line.strip('\n'))

        if type(string) is str:
            string = jieba.cut(string)

        new_string = list()
        for word in string:
            if word in stopwords:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result

def remove_less_word(lines_str, word_st):
    return " ".join([word for word in lines_str.split() if word in word_st])

def remove_single_nodepost(p_idx, p_node, k_idx, k_node, y, p_root_idx):
    p_idx_new = []
    p_node_new = []
    k_idx_new = []
    k_node_new = []
    y_new = []
    p_root_idx_new = []
    for i in range(len(p_idx)):
        if len(p_idx[i]) > 1:
            p_idx_new.append(p_idx[i])
            p_node_new.append(p_node[i])
            k_idx_new.append(k_idx[i])
            k_node_new.append(k_node[i])
            y_new.append(y[i])
            p_root_idx_new.append(p_root_idx[i])
    p_idx_new = np.array(p_idx_new)
    p_node_new = np.array(p_node_new)
    k_idx_new = np.array(k_idx_new)
    k_node_new = np.array(k_node_new)
    y_new = np.array(y_new)
    p_root_idx_new = np.array(p_root_idx_new)
    return p_idx_new,p_node_new,k_idx_new,k_node_new,y_new, p_root_idx_new

class data_process_nn():
    def __init__(self,sen_len):
        self.sen_len = sen_len
        self.node2idx_dict = {}
        self.idx2node_dict = {}
        self.mid2text_dict = {}
        self.mid2regtextidx_dict = {}
        self.token_dict = {}
        self.token_lst = []

    def get_node2id(self):
        with open(path_node2idx_mid, 'r', encoding='utf-8') as f:
            node2idx_mid = f.readlines()
        for line in node2idx_mid:
            node = line.strip('\n').split('\t')[0]
            idx = int(line.strip('\n').split('\t')[1])
            self.idx2node_dict[idx] = node
            self.node2idx_dict[node] = idx

    def get_mid2text(self):
        with open(path_mid2text,'r',encoding='utf-8') as f:
            mid2text = f.readlines()
        for line in mid2text:
            mid = line.split('\t')[0]
            text = line.split('\t')[1]
            self.mid2text_dict[mid] = text

    def get_token(self):
        string_process = StringProcess()
        for mid in self.mid2text_dict:
            mid_idx = self.node2idx_dict[mid]
            textidx = []
            text = self.mid2text_dict[mid]
            text = string_process.clean_str(text)
            for word in text.split(' '):
                if word not in self.token_dict:
                    self.token_dict[word] = len(self.token_lst)
                    self.token_lst.append(word)
                textidx.append(self.token_dict[word])
            self.mid2regtextidx_dict[mid_idx] = textidx
        for idx in self.idx2node_dict:
            if self.idx2node_dict[idx] not in self.mid2text_dict:
                token = self.idx2node_dict[idx]
                self.token_dict[idx] = len(self.token_lst)
                self.token_lst.append(token)

        for mid_idx_ in self.mid2regtextidx_dict:
            textidx = self.pad_sequence(self.mid2regtextidx_dict[mid_idx_],self.sen_len)
            self.mid2regtextidx_dict[mid_idx_] = textidx

    def load_sparse_temporal_data(self,train, val, test):
        self.get_node2id()
        self.get_mid2text()
        self.get_token()
        names = ['propagation_node_idx.npy', 'propagation_node.npy', 'knowledge_node_idx.npy', 'knowledge_node.npy',
                 'label.npy','propagation_root_index.npy']
        objects = []
        for i in range(len(names)):
            with open(path_temporal + "/{}".format(names[i]), 'rb') as f:
                objects.append(np.load(f, encoding='latin1', allow_pickle=True))
        p_idx, p_node, k_idx, k_node, y,p_root_idx = tuple(objects)
        p_idx, p_node, k_idx, k_node, y ,p_root_idx= remove_single_nodepost(p_idx, p_node, k_idx, k_node, y, p_root_idx)
        for xx in p_node:
            for i in range(len(xx)):
                xx[i] = sparse_mx_to_torch_sparse_tensor(normalize_adj(xx[i]))

        y = torch.from_numpy(y).float()
        y = torch.unsqueeze(y, 1)
        p_root_idx = [_idx.astype(int) for _idx in p_root_idx[:]]
        p_root_idx = np.array(p_root_idx)
        p_root_idx = torch.from_numpy(p_root_idx).long()
        p_root_idx = torch.unsqueeze(p_root_idx,1)

        p_idx = [_idx.astype(int) for _idx in p_idx[:]]
        p_idx = [torch.from_numpy(_idx).long() for _idx in p_idx[:]]

        for xx in k_node:
            for i in range(len(xx)):
                xx[i] = sparse_mx_to_torch_sparse_tensor(normalize_adj(xx[i]))
        k_idx = [_idx.astype(int) for _idx in k_idx[:]]
        k_idx = [torch.from_numpy(_idx).long() for _idx in k_idx[:]]

        train_idx, val_idx, test_idx = split_data(len(p_idx), train, val, test, shuffle=False)

        train_dict, val_dict, test_dict = {}, {}, {}

        names_dict = {'x_p': p_node, 'x_k': k_node, 'y': y, 'idx_p': p_idx, 'idx_k': k_idx, 'root_idx_p':p_root_idx}
        for name in names_dict:
            train_dict[name] = [names_dict[name][i] for i in train_idx]
            val_dict[name] = [names_dict[name][i] for i in val_idx]
            test_dict[name] = [names_dict[name][i] for i in test_idx]

        return train_dict, val_dict, test_dict, self.token_dict, self.mid2regtextidx_dict

    def pad_sequence(self,text,max_length):
        if len(text) > max_length:
            text = text[:max_length]
        elif len(text) < max_length:
            pad_index = len(self.token_lst)
            if not pad_index in self.token_dict:
                self.token_dict[pad_index] = pad_index
            for i in range(max_length - len(text)):
                text.append(pad_index)
        assert len(text) == max_length
        return text

class data_process_nn_nBatch():
    def __init__(self,sen_len,pathset,dataset):
        self.sen_len = sen_len
        self.pathset = pathset
        self.dataset = dataset
        self.node2idx_dict = {}
        self.idx2node_dict = {}
        self.mid2text_dict = {}
        self.mid2regtextidx_dict = {}
        self.token_dict = {}
        self.token_lst = []

    def get_node2id(self):
        with open(self.pathset.path_node2idx_mid, 'r', encoding='utf-8') as f:
            node2idx_mid = f.readlines()
        for line in node2idx_mid:
            node = line.strip('\n').split('\t')[0]
            idx = int(line.strip('\n').split('\t')[1])
            self.idx2node_dict[idx] = node
            self.node2idx_dict[node] = idx

    def get_mid2text(self):
        with open(self.pathset.path_mid2text,'r',encoding='utf-8') as f:
            mid2text = f.readlines()
        for line in mid2text:
            mid = line.strip('\n').split('\t')[0]
            text = line.strip('\n').split('\t')[1]
            self.mid2text_dict[mid] = text

    def get_token(self):
        string_process = StringProcess()
        if self.dataset == 'pheme':
            for mid in self.mid2text_dict:
                mid_idx = self.node2idx_dict[mid]
                textidx = []
                text = self.mid2text_dict[mid]
                text = string_process.clean_str(text)
                for word in text.split(' '):
                    if word not in self.token_dict:
                        self.token_dict[word] = len(self.token_lst)
                        self.token_lst.append(word)
                    textidx.append(self.token_dict[word])
                self.mid2regtextidx_dict[mid_idx] = textidx
            for idx in self.idx2node_dict:
                if self.idx2node_dict[idx] not in self.mid2text_dict:
                    token = self.idx2node_dict[idx]
                    self.token_dict[idx] = len(self.token_lst)
                    self.token_lst.append(token)
            for mid_idx_ in self.mid2regtextidx_dict:
                textidx = self.pad_sequence(self.mid2regtextidx_dict[mid_idx_],self.sen_len)
                self.mid2regtextidx_dict[mid_idx_] = textidx
            print('data process node2idx_dict:',len(self.node2idx_dict))
            print('data process token_lst:',len(self.token_lst))
            print('data process token_dict:', len(self.token_dict))


        if self.dataset == 'weibo':
            for mid in self.mid2text_dict:
                mid_idx = self.node2idx_dict[mid]
                textidx = []
                text = self.mid2text_dict[mid]
                text = string_process.clean_str_zh(text)
                for word in text.split(' '):
                    if word not in self.token_dict:
                        self.token_dict[word] = len(self.token_lst)
                        self.token_lst.append(word)
                    textidx.append(self.token_dict[word])
                self.mid2regtextidx_dict[mid_idx] = textidx
            for idx in self.idx2node_dict:
                if self.idx2node_dict[idx] not in self.mid2text_dict:
                    token = self.idx2node_dict[idx]
                    self.token_dict[idx] = len(self.token_lst)
                    self.token_lst.append(token)
            for mid_idx_ in self.mid2regtextidx_dict:
                textidx = self.pad_sequence(self.mid2regtextidx_dict[mid_idx_],self.sen_len)
                self.mid2regtextidx_dict[mid_idx_] = textidx


    def load_sparse_temporal_data(self,train, val, test):
        self.get_node2id()
        self.get_mid2text()
        self.get_token()
        names = ['propagation_node_idx.npy', 'propagation_node.npy', 'knowledge_node_idx.npy', 'knowledge_node.npy',
                 'label.npy','propagation_root_index.npy']
        objects = []
        for i in range(len(names)):
            with open(self.pathset.path_temporal + "/{}".format(names[i]), 'rb') as f:
                objects.append(np.load(f, encoding='latin1', allow_pickle=True))
        p_idx, p_node, k_idx, k_node, y,p_root_idx = tuple(objects)
        print('data process p_idx:',len(p_idx))
        print('data process p_root_idx:',len(p_root_idx))
        p_idx, p_node, k_idx, k_node, y ,p_root_idx= remove_single_nodepost(p_idx, p_node, k_idx, k_node, y, p_root_idx)
        p_node_indices = []
        p_node_values = []
        for xx in p_node:
            xx_indices, xx_values = [],[]
            for i in range(len(xx)):
                indices, values, shape = sparse_mx_to_torch(normalize_adj(xx[i]))

                xx_indices.append(indices)
                xx_values.append(values)

            p_node_indices.append(xx_indices)
            p_node_values.append(xx_values)

        y = torch.from_numpy(y).long()
        y = torch.unsqueeze(y, 1)

        p_root_idx = [_idx.astype(int) for _idx in p_root_idx[:]]
        p_root_idx = np.array(p_root_idx)
        p_root_idx = torch.from_numpy(p_root_idx).long()
        p_root_idx = torch.unsqueeze(p_root_idx,1)

        p_idx = [_idx.astype(int) for _idx in p_idx[:]]
        p_idx = [torch.from_numpy(_idx).long() for _idx in p_idx[:]]

        k_node_indices, k_node_values = [], []
        for xx in k_node:
            xx_indices,xx_values = [],[]
            for i in range(len(xx)):
                indices, values, shape= sparse_mx_to_torch(normalize_adj(xx[i]))
                xx_indices.append(indices)
                xx_values.append(values)
            k_node_indices.append(xx_indices)
            k_node_values.append(xx_values)

        k_idx = [_idx.astype(int) for _idx in k_idx[:]]
        k_idx = [torch.from_numpy(_idx).long() for _idx in k_idx[:]]

        train_idx, val_idx, test_idx = split_data(len(p_idx),y, train, val, test, shuffle=False)

        train_dict, val_dict, test_dict = {}, {}, {}

        names_dict = {'x_p_indices': p_node_indices, 'x_p_values':p_node_values, 'x_k_indices': k_node_indices,'x_k_values':k_node_values,\
                      'y': y, 'idx_p': p_idx, 'idx_k': k_idx, 'root_idx_p':p_root_idx}
        for name in names_dict:
            train_dict[name] = [names_dict[name][i] for i in train_idx]
            val_dict[name] = [names_dict[name][i] for i in val_idx]
            test_dict[name] = [names_dict[name][i] for i in test_idx]

        return train_dict, val_dict, test_dict, self.token_dict, self.mid2regtextidx_dict

    def pad_sequence(self,text,max_length):
        if len(text) > max_length:
            text = text[:max_length]
        elif len(text) < max_length:
            pad_index = len(self.token_lst)
            if not '<pad_index>' in self.token_dict:
                self.token_dict['<pad_index>'] = pad_index
            for i in range(max_length - len(text)):
                text.append(pad_index)
        assert len(text) == max_length
        return text

class data_process_nn_nBatch_BERT():
    def __init__(self,sen_len,pathset,dataset):
        self.sen_len = sen_len
        self.pathset = pathset
        self.dataset = dataset
        self.node2idx_dict = {}
        self.idx2node_dict = {}
        self.mid2text_dict = {}
        self.mid2bert_tokenizer = {}
        UNCASED = self.pathset.path_bert
        VOCAB = self.pathset.VOCAB
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED,VOCAB))

    def get_node2id(self):
        with open(self.pathset.path_node2idx_mid, 'r', encoding='utf-8') as f:
            node2idx_mid = f.readlines()
        for line in node2idx_mid:
            node = line.strip('\n').split('\t')[0]
            idx = int(line.strip('\n').split('\t')[1])
            self.idx2node_dict[idx] = node
            self.node2idx_dict[node] = idx

    def get_mid2text(self):
        with open(self.pathset.path_mid2text,'r',encoding='utf-8') as f:
            mid2text = f.readlines()
        for line in mid2text:
            mid = line.strip('\n').split('\t')[0]
            text = line.strip('\n').split('\t')[1]
            self.mid2text_dict[mid] = text

    def get_token(self):
        string_process = StringProcess()
        for mid in self.mid2text_dict:
            mid_idx = self.node2idx_dict[mid]
            textidx = []
            text = self.mid2text_dict[mid]
            text = string_process.clean_str_BERT(text)
            tokenizer_encoding = self.tokenizer(text, return_tensors='pt', padding='max_length',\
                                                truncation=True,max_length=self.sen_len)
            self.mid2bert_tokenizer[mid_idx] = tokenizer_encoding

        print('data process node2idx_dict:',len(self.node2idx_dict))


    def load_sparse_temporal_data(self,train, val, test):
        self.get_node2id()
        self.get_mid2text()
        self.get_token()
        names = ['propagation_node_idx.npy', 'propagation_node.npy', 'knowledge_node_idx.npy', 'knowledge_node.npy',
                 'label.npy','propagation_root_index.npy']
        objects = []
        for i in range(len(names)):
            with open(self.pathset.path_temporal + "/{}".format(names[i]), 'rb') as f:
                objects.append(np.load(f, encoding='latin1', allow_pickle=True))
        p_idx, p_node, k_idx, k_node, y,p_root_idx = tuple(objects)
        print('data process p_idx:',len(p_idx))
        print('data process p_root_idx:',len(p_root_idx))
        p_idx, p_node, k_idx, k_node, y ,p_root_idx= remove_single_nodepost(p_idx, p_node, k_idx, k_node, y, p_root_idx)
        p_node_indices = []
        p_node_values = []
        for xx in p_node:
            xx_indices, xx_values = [],[]
            for i in range(len(xx)):
                indices, values, shape = sparse_mx_to_torch(normalize_adj(xx[i]))

                xx_indices.append(indices)
                xx_values.append(values)

            p_node_indices.append(xx_indices)
            p_node_values.append(xx_values)

        y = torch.from_numpy(y).long()
        y = torch.unsqueeze(y, 1)

        p_root_idx = [_idx.astype(int) for _idx in p_root_idx[:]]
        p_root_idx = np.array(p_root_idx)
        p_root_idx = torch.from_numpy(p_root_idx).long()
        p_root_idx = torch.unsqueeze(p_root_idx,1)

        p_idx = [_idx.astype(int) for _idx in p_idx[:]]
        p_idx = [torch.from_numpy(_idx).long() for _idx in p_idx[:]]

        k_node_indices, k_node_values = [], []

        for xx in k_node:
            xx_indices,xx_values = [],[]
            for i in range(len(xx)):
                indices, values, shape= sparse_mx_to_torch(normalize_adj(xx[i]))
                xx_indices.append(indices)
                xx_values.append(values)

            k_node_indices.append(xx_indices)
            k_node_values.append(xx_values)

        k_idx = [_idx.astype(int) for _idx in k_idx[:]]
        k_idx = [torch.from_numpy(_idx).long() for _idx in k_idx[:]]

        train_idx, val_idx, test_idx = split_data(len(p_idx),y, train, val, test, shuffle=True)

        train_dict, val_dict, test_dict = {}, {}, {}

        names_dict = {'x_p_indices': p_node_indices, 'x_p_values':p_node_values, 'x_k_indices': k_node_indices,'x_k_values':k_node_values,\
                      'y': y, 'idx_p': p_idx, 'idx_k': k_idx, 'root_idx_p':p_root_idx}
        for name in names_dict:
            train_dict[name] = [names_dict[name][i] for i in train_idx]
            val_dict[name] = [names_dict[name][i] for i in val_idx]
            test_dict[name] = [names_dict[name][i] for i in test_idx]

        return train_dict, val_dict, test_dict, self.node2idx_dict, self.mid2bert_tokenizer

    def pad_sequence(self,text,max_length):
        if len(text) > max_length:
            text = text[:max_length]
        elif len(text) < max_length:
            pad_index = len(self.token_lst)
            if not '<pad_index>' in self.token_dict:
                self.token_dict['<pad_index>'] = pad_index
            for i in range(max_length - len(text)):
                text.append(pad_index)
        assert len(text) == max_length
        return text

def loadData_bert(x_propagation_idx,x_propagation_node_indices, x_propagation_node_values, \
             x_knowledge_idx,x_knowledge_node_indices,x_knowledge_node_values,target, root_idx):
    data_list = GraphDataset_bert(x_propagation_idx,x_propagation_node_indices, x_propagation_node_values, \
                             x_knowledge_idx,x_knowledge_node_indices,x_knowledge_node_values,\
                             target, root_idx)
    return data_list


class GraphDataset_bert(Dataset):
    def __init__(self, x_propagation_idx,x_propagation_node_indices,x_propagation_node_values,\
                 x_knowledge_idx,x_knowledge_node_indices,x_knowledge_node_values,target, root_idx):
        self.x_propagation_idx = x_propagation_idx
        self.x_propagation_node_indices = x_propagation_node_indices
        self.x_propagation_node_values = x_propagation_node_values
        self.x_knowledge_idx = x_knowledge_idx
        self.x_knowledge_node_indices = x_knowledge_node_indices
        self.x_knowledge_node_values = x_knowledge_node_values
        self.target = target
        self.root_idx = root_idx

    def __len__(self):
        return len(self.target)


    def __getitem__(self, index):
        #==========================propogation graph==============================
        x_propagation_node_num = len(self.x_propagation_idx[index])
        x_propagation_node_idx = self.x_propagation_idx[index]
        x_propagation_node_idx = x_propagation_node_idx.unsqueeze(1)
        #--------------progation_graph_0----------------
        x_propagation_node_indices_0 = self.x_propagation_node_indices[index][0]
        x_p_edge_num_0 = self.x_propagation_node_indices[index][0].size(1)
        x_propagation_node_values_0 = self.x_propagation_node_values[index][0]

        x_propagation_node_values_0 = torch.tensor(x_propagation_node_values_0, dtype=torch.float)

        #-------------propagation_graph_1---------------
        x_propagation_node_indices_1 = self.x_propagation_node_indices[index][1]
        x_p_edge_num_1 = self.x_propagation_node_indices[index][1].size(1)
        x_propagation_node_values_1 = self.x_propagation_node_values[index][1]

        #-------------propagtion_graph_2----------------
        x_propagation_node_indices_2 = self.x_propagation_node_indices[index][2]
        x_p_edge_num_2 = self.x_propagation_node_indices[index][2].size(1)
        x_propagation_node_values_2 = self.x_propagation_node_values[index][2]

        #=================================knowledge_graph================================
        x_knowledge_node_num = len(self.x_knowledge_idx[index])
        x_knowledge_node_idx = self.x_knowledge_idx[index]
        x_knowledge_node_idx = x_knowledge_node_idx.unsqueeze(1)
        #------------knowledge_graph_0------------------
        x_knowledge_node_indices_0 = self.x_knowledge_node_indices[index][0]
        x_k_edge_num_0 = self.x_knowledge_node_indices[index][0].size(1)
        x_knowledge_node_values_0 = self.x_knowledge_node_values[index][0]
        #------------knowledge_graph_1------------------
        x_knowledge_node_indices_1 = self.x_knowledge_node_indices[index][1]
        x_k_edge_num_1 = self.x_knowledge_node_indices[index][1].size(1)
        x_knowledge_node_values_1 = self.x_knowledge_node_values[index][1]
        #------------knowledge_graph_2------------------
        x_knowledge_node_indices_2 = self.x_knowledge_node_indices[index][2]
        x_k_edge_num_2 = self.x_knowledge_node_indices[index][2].size(1)
        x_knowledge_node_values_2 = self.x_knowledge_node_values[index][2]

        if self.target[index].item() == 0:
            y = torch.tensor([1,0]).unsqueeze(0)
        elif self.target[index].item() == 1:
            y = torch.tensor([0,1]).unsqueeze(0)

        return Data(
                    x = torch.tensor(x_propagation_node_idx,dtype=torch.long),\
                    edge_index = torch.LongTensor(x_propagation_node_indices_0), \
                    edge_values=torch.tensor(x_propagation_node_values_0, dtype=torch.float), \
                    edge_num=torch.tensor([x_p_edge_num_0], dtype=torch.long), \

                    x_propagation_node_indices_1_edge_index= torch.LongTensor(x_propagation_node_indices_1), \
                    x_propagation_node_values_1=torch.tensor(x_propagation_node_values_1, dtype=torch.float), \
                    x_propagation_edge_num_1=torch.tensor([x_p_edge_num_1], dtype=torch.long), \

                    x_propagation_node_indices_2_edge_index = torch.LongTensor(x_propagation_node_indices_2), \
                    x_propagation_node_values_2 = torch.tensor(x_propagation_node_values_2,dtype=torch.float),\
                    x_propagation_edge_num_2 = torch.tensor([x_p_edge_num_2],dtype=torch.long), \

                    x_propagation_node_num = torch.tensor([x_propagation_node_num],dtype=torch.long),\
                    #=============================================================================
                    x_knowledge_idx = torch.tensor(x_knowledge_node_idx,dtype=torch.long),\
                    x_knowledge_node_indices_0_edge_index = torch.LongTensor(x_knowledge_node_indices_0), \
                    x_knowledge_node_values_0=torch.tensor(x_knowledge_node_values_0, dtype=torch.float), \
                    x_knowledge_edge_num_0=torch.tensor([x_k_edge_num_0], dtype=torch.long), \

                    x_knowledge_node_indices_1_edge_index = torch.LongTensor(x_knowledge_node_indices_1), \
                    x_knowledge_node_values_1=torch.tensor(x_knowledge_node_values_1, dtype=torch.float), \
                    x_knowledge_edge_num_1=torch.tensor([x_k_edge_num_1], dtype=torch.long), \

                    x_knowledge_node_indices_2_edge_index = torch.LongTensor(x_knowledge_node_indices_2), \
                    x_knowledge_node_values_2 = torch.tensor(x_knowledge_node_values_2,dtype=torch.float), \
                    x_knowledge_edge_num_2=torch.tensor([x_k_edge_num_2], dtype=torch.long), \

                    x_knowledge_node_num = torch.tensor([x_knowledge_node_num],dtype=torch.long),\
                    #============================================================================
                    target = torch.tensor(y,dtype=torch.float),\
                    root_idx = torch.LongTensor(self.root_idx[index]))
