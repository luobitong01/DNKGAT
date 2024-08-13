import os
import pandas as pd
import datetime as dt
from dateutil import parser
from dateutil import rrule
import re
import numpy as np
from math import log
import itertools
from collections import defaultdict
from tqdm import tqdm
from scipy import sparse
import csv

pheme_clean_path = '../data/pheme/pheme_clean/'
pheme_entity_path = '../data/pheme/pheme_entity/'
pheme_concept_yago_path = '../data/pheme/pheme_concept_yago/'
pheme_temporal_path = '../data/pheme/pheme_temporal_data/'

def month_trans(mon):
    mon_dic = {}
    mon_dic['Jan'] = 1
    mon_dic['Feb'] = 2
    mon_dic['Mar'] = 3
    mon_dic['Apr'] = 4
    mon_dic['May'] = 5
    mon_dic['Jun'] = 6
    mon_dic['Jul'] = 7
    mon_dic['Aug'] = 8
    mon_dic['Sep'] = 9
    mon_dic['Oct'] = 10
    mon_dic['Nov'] = 11
    mon_dic['Dec'] = 12
    return mon_dic[mon]
    
def trans_time(t, t_init):
    t = t.split(' ')
    t_exct = t[3].split(':')
    t_init = t_init.split(' ')
    t_init_exct = t_init[3].split(':')
    date_1 = dt.datetime(int(t[5]),month_trans(t[1]),int(t[2]),int(t_exct[0]),int(t_exct[1]),int(t_exct[2]))
    date_0 = dt.datetime(int(t_init[5]), month_trans(t_init[1]), int(t_init[2]), int(t_init_exct[0]), int(t_init_exct[1]), int(t_init_exct[2]))
    interval = (date_1 - date_0).seconds
    return interval
    
def clean_text_en(text):
    r1 = "\n"
    r2 = '\r'
    r3 = '\t'
    text = re.sub(r1,' ',text)
    text = re.sub(r2,' ',text)
    text = re.sub(r3,' ',text)
    # text = re.sub(r1,'',text)
    return text

def read_data():
    tree_dic = {}
    pheme_files_entity = os.listdir(pheme_entity_path)
    for i in range(len(pheme_files_entity)):
        file = pheme_files_entity[i].split('.')[0]
        tree_dic[file] = {}
        file_df_entity = pd.read_csv(pheme_entity_path + file + '.csv')
        j_init = 0
        for j in range(len(file_df_entity['mid'])):
            if file_df_entity['parent'][j] == 'None':
                j_init = j
        t_init = file_df_entity['t'][j_init]
        for j in range(len(file_df_entity['mid'])):
            mid = file_df_entity['mid'][j]
            if not mid in tree_dic[file]:
                tree_dic[file][mid] = {}
                tree_dic[file][mid]['parent'] = file_df_entity['parent'][j].strip('\t')
                tree_dic[file][mid]['text'] = clean_text_en(file_df_entity['text'][j])
                t_trans = trans_time(file_df_entity['t'][j], t_init)
                tree_dic[file][mid]['t'] = t_trans
                entity = [ent.strip(' ').strip('\'').strip(' ') for ent in file_df_entity['entity'][j].strip('\t').strip('[').strip(']').split(',')]
                for ent in entity:
                    if len(ent) == 0:
                        entity.remove(ent)
                tree_dic[file][mid]['entity'] = entity

    pheme_files_concept = os.listdir(pheme_concept_yago_path)
    for i in range(len(pheme_files_concept)):
        file = pheme_files_concept[i].split('.')[0]
        file_df_concept = pd.read_csv(pheme_concept_yago_path + file + '.csv')
        for j in range(len(file_df_concept['mid'])):
            mid = file_df_concept['mid'][j]
            concept = [(' ').join(conc.strip(' ').strip('\'').strip('<').strip('>').split('_')) for conc in file_df_concept['concept'][j].strip('\t').strip('[').strip(']').split(',')]
            concept = [con.strip(' ') for con in concept]
            for con in concept:
                if len(con) == 0:
                    concept.remove(con)
            tree_dic[file][mid]['concept'] = concept
    return tree_dic

def time_equal_segment(sub_tree_dic):
    t_list = []
    for mid in sub_tree_dic:
        t = sub_tree_dic[mid]['t']
        t_list.append(t)
    max_t = max(t_list)
    sliding_T = max_t/3
    T_num = 3
    return sliding_T, T_num

def node2index():
    tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(pheme_entity_path)]
    node_lst = []

    for file in files_name:
        for mid in tree_dic[file]:
            node_lst.append(str(mid))
    for file in files_name:
        for mid in tree_dic[file]:
            node_lst += tree_dic[file][mid]['entity']
            node_lst += tree_dic[file][mid]['concept']
    node_lst = list(set(node_lst))
    with open('../data/pheme/' + 'node2idx_mid.txt', 'w', encoding='utf-8', newline='')as f:
        for i, node in enumerate(node_lst):
            string = node + '\t' + str(i) + '\n'
            f.writelines(string)
    with open('../data/pheme/'+'mid2text.txt','w',encoding='utf-8',newline='')as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid) +'\t' + tree_dic[file][mid]['text'] + '\n'
                f.writelines(string)

def load_node2index():
    with open('../data/pheme/' + 'node2idx_mid.txt', 'r',encoding='utf-8')as f:
        node2idx = f.readlines()
    node2idx_dict = {}
    for line in node2idx:
        node = line.strip('\n').split('\t')[0]
        idx = line.strip('\n').split('\t')[1]
        node2idx_dict[node] = idx
    return node2idx_dict
    
def process_node2index(node2idx, sub_tree_dic):
    node_lst = []
    text_node_lst = []
    for mid in sub_tree_dic:
        mid_new = str(mid)
        node_lst.append(node2idx[mid_new])
        text_node_lst.append(node2idx[mid_new])
        for entity in sub_tree_dic[mid]['entity']:
            node_lst.append(node2idx[entity])
        for concept in sub_tree_dic[mid]['concept']:
            node_lst.append(node2idx[concept])
    node_lst = list(set(node_lst))
    return node_lst, text_node_lst

#--------------pmi------------------------------------
def get_window(content_lst,window_size):
    word_window_freq = defaultdict(int)  # w(i) 
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    windows = list()
    length = len(content_lst)

    if length <= window_size:
        windows.append(content_lst)
    else:
        for j in range(length - window_size + 1):
            window = content_lst[j: j + window_size]
            windows.append(list(set(window)))

    for window in windows:
        for word in window:
            word_window_freq[word] += 1

        for word_pair in itertools.combinations(window, 2):
            word_pair_count[word_pair] += 1

    windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst

def get_pmi_edge(content_lst,node_idx, sub_tree_dic, window_size=20, threshold=0.):
    word_window_freq, word_pair_count, windows_len = get_window(content_lst,window_size)
    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    return pmi_edge_lst

#-------------tf-idf----------------------------------
def get_tfidf_edge(entity, entity_list, sub_tree_dic):
    tf = get_tf_score(entity,entity_list)
    idf = get_idf_score(entity, sub_tree_dic)
    return tf*idf

def get_tf_score(entity, entity_list):
    tf = entity_list.count(entity)/len(entity_list)
    return tf

def get_idf_score(entity, sub_tree_dic):
    count = 0
    for mid in sub_tree_dic:
        if entity in sub_tree_dic[mid]['entity']:
            count += 1
    idf = log(len(sub_tree_dic)/(count + 1))
    return idf
#------------------------------------------------------
def build_temporal_propagation_graph(text_node_idx,node2idx,sub_tree_dic):
    length = len(text_node_idx)
    sliding_T, T_num = time_equal_segment(sub_tree_dic)
    temporal_matrix = np.zeros((T_num,length,length),dtype=np.float)
    for mid in sub_tree_dic:
        if sub_tree_dic[mid]['t'] < sliding_T:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                assert new_idx != new_parent_idx
                temporal_matrix[0][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[0][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[1][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[1][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)
        elif sub_tree_dic[mid]['t'] < 2*sliding_T and sub_tree_dic[mid]['t'] >= sliding_T:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                assert new_idx != new_parent_idx
                temporal_matrix[1][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[1][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)
        else:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                assert new_idx != new_parent_idx
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)
    return temporal_matrix

def build_temporal_knowledge_graph(node_idx,node2idx,sub_tree_dic):
    length = len(node_idx)
    sliding_T, T_num = time_equal_segment(sub_tree_dic)
    temporal_matrix = np.zeros((T_num, length, length),dtype=np.float)
    content_lst_T = []
    content_lst_2T = []
    content_lst_3T = []
    #--------tfidf-----------------
    for mid in sub_tree_dic:
        # for i in range(T_num):
        if sub_tree_dic[mid]['t'] < sliding_T:
            idx = node2idx[str(mid)]
            new_idx = node_idx.index(idx)
            entity_lst = sub_tree_dic[mid]['entity']
            concept_lst = sub_tree_dic[mid]['concept']
            for entity in entity_lst:
                entity_idx = node2idx[entity]
                new_entity_idx = node_idx.index(entity_idx)
                assert new_idx != new_entity_idx
                tfidf = get_tfidf_edge(entity, entity_lst, sub_tree_dic)
                temporal_matrix[0][new_idx][new_entity_idx] = tfidf
                temporal_matrix[0][new_entity_idx][new_idx] = tfidf
                temporal_matrix[1][new_idx][new_entity_idx] = tfidf
                temporal_matrix[1][new_entity_idx][new_idx] = tfidf
                temporal_matrix[2][new_idx][new_entity_idx] = tfidf
                temporal_matrix[2][new_entity_idx][new_idx] = tfidf
                content_lst_T.append(entity)
            for concept in concept_lst:
                content_lst_T.append(concept)
        elif sub_tree_dic[mid]['t'] < 2*sliding_T and sub_tree_dic[mid]['t'] >= sliding_T:
            idx = node2idx[str(mid)]
            new_idx = node_idx.index(idx)
            entity_lst = sub_tree_dic[mid]['entity']
            concept_lst = sub_tree_dic[mid]['concept']
            for entity in entity_lst:
                entity_idx = node2idx[entity]
                new_entity_idx = node_idx.index(entity_idx)
                assert new_idx != new_entity_idx
                tfidf = get_tfidf_edge(entity, entity_lst, sub_tree_dic)
                temporal_matrix[1][new_idx][new_entity_idx] = tfidf
                temporal_matrix[1][new_entity_idx][new_idx] = tfidf
                temporal_matrix[2][new_idx][new_entity_idx] = tfidf
                temporal_matrix[2][new_entity_idx][new_idx] = tfidf
                content_lst_2T.append(entity)
            for concept in concept_lst:
                content_lst_2T.append(concept)
        else:
            idx = node2idx[str(mid)]
            new_idx = node_idx.index(idx)
            entity_lst = sub_tree_dic[mid]['entity']
            concept_lst = sub_tree_dic[mid]['concept']
            for entity in entity_lst:
                entity_idx = node2idx[entity]
                new_entity_idx = node_idx.index(entity_idx)
                tfidf = get_tfidf_edge(entity, entity_lst, sub_tree_dic)
                temporal_matrix[2][new_idx][new_entity_idx] = tfidf
                temporal_matrix[2][new_entity_idx][new_idx] = tfidf
                content_lst_3T.append(entity)
            for concept in concept_lst:
                content_lst_3T.append(concept)

    pmi_edge_list_T = get_pmi_edge(content_lst_T, node_idx, sub_tree_dic, window_size=20, threshold=0.)
    pmi_edge_list_2T = get_pmi_edge(content_lst_2T, node_idx, sub_tree_dic, window_size=20, threshold=0.)
    pmi_edge_list_3T = get_pmi_edge(content_lst_3T, node_idx, sub_tree_dic, window_size=20, threshold=0.)
    for word_pair_0, word_pair_1, pmi in pmi_edge_list_T:
        idx_0 = node2idx[word_pair_0]
        new_idx_0 = node_idx.index(idx_0)
        idx_1 = node2idx[word_pair_1]
        new_idx_1 = node_idx.index(idx_1)
        assert new_idx_0 != new_idx_1
        temporal_matrix[0][new_idx_0][new_idx_1] = pmi
        temporal_matrix[0][new_idx_1][new_idx_0] = pmi
        temporal_matrix[1][new_idx_0][new_idx_1] = pmi
        temporal_matrix[1][new_idx_1][new_idx_0] = pmi
        temporal_matrix[2][new_idx_0][new_idx_1] = pmi
        temporal_matrix[2][new_idx_1][new_idx_0] = pmi
    for word_pair_0, word_pair_1, pmi in pmi_edge_list_2T:
        idx_0 = node2idx[word_pair_0]
        new_idx_0 = node_idx.index(idx_0)
        idx_1 = node2idx[word_pair_1]
        new_idx_1 = node_idx.index(idx_1)
        assert new_idx_0 != new_idx_1
        temporal_matrix[1][new_idx_0][new_idx_1] = pmi
        temporal_matrix[1][new_idx_1][new_idx_0] = pmi
        temporal_matrix[2][new_idx_0][new_idx_1] = pmi
        temporal_matrix[2][new_idx_1][new_idx_0] = pmi
    for word_pair_0, word_pair_1, pmi in pmi_edge_list_3T:
        idx_0 = node2idx[word_pair_0]
        new_idx_0 = node_idx.index(idx_0)
        idx_1 = node2idx[word_pair_1]
        new_idx_1 = node_idx.index(idx_1)
        assert new_idx_0 != new_idx_1
        temporal_matrix[2][new_idx_0][new_idx_1] = pmi
        temporal_matrix[2][new_idx_1][new_idx_0] = pmi
    return temporal_matrix
    
def load_mid2label():
    label_path = '../data/pheme/pheme_id_label.txt'
    mid_label = list(open(label_path, "r",encoding='utf-8'))
    mid2label_dict = {}
    for m_l in mid_label:
        mid = m_l.strip('\n').split('\t')[0]
        label = m_l.strip('\n').split('\t')[1]
        mid2label_dict[mid] = int(label)
    return mid2label_dict

def main():
    node2idx_dict = load_node2index()
    mid2label_dict = load_mid2label()
    tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(pheme_entity_path)]
    node_idx_final = []
    text_node_idx_final = []
    temp_propagation_graph_final = []
    temp_knowledge_graph_final = []
    label_final = []
    root_index_final = []
    for file in files_name:
        root_index_final.append(node2idx_dict[file])
        label_final.append(mid2label_dict[file])
        node_idx, text_node_idx = process_node2index(node2idx_dict,tree_dic[file])

        temp_propagation_graph = build_temporal_propagation_graph(text_node_idx,node2idx_dict,tree_dic[file])
        temp_knowledge_graph = build_temporal_knowledge_graph(node_idx,node2idx_dict,tree_dic[file])
        print('propagation graph',temp_propagation_graph.shape)
        print('knowledge graph', temp_knowledge_graph.shape)

        text_node_idx = np.array(text_node_idx)
        node_idx = np.array(node_idx)
        node_idx_final.append(node_idx)
        text_node_idx_final.append(text_node_idx)
        propagation_s0 = sparse.csr_matrix(temp_propagation_graph[0])
        propagation_s1 = sparse.csr_matrix(temp_propagation_graph[1])
        propagation_s2 = sparse.csr_matrix(temp_propagation_graph[2])
        temp_propagation_graph_final.append([propagation_s0,propagation_s1,propagation_s2])

        knowledge_s0 = sparse.csr_matrix(temp_knowledge_graph[0])
        knowledge_s1 = sparse.csr_matrix(temp_knowledge_graph[1])
        knowledge_s2 = sparse.csr_matrix(temp_knowledge_graph[2])
        temp_knowledge_graph_final.append([knowledge_s0,knowledge_s1,knowledge_s2])

    with open(pheme_temporal_path+'propagation_node_idx.npy','wb')as f:
        text_node_idx_final = np.array(text_node_idx_final)
        np.save(f,text_node_idx_final)
    with open(pheme_temporal_path+'knowledge_node_idx.npy','wb')as f:
        node_idx_final = np.array(node_idx_final)
        np.save(f,node_idx_final)
    with open(pheme_temporal_path+'propagation_node.npy','wb')as f:
        temp_propagation_graph_final = np.array(temp_propagation_graph_final)
        np.save(f, temp_propagation_graph_final)
    with open(pheme_temporal_path+'knowledge_node.npy','wb')as f:
        temp_knowledge_graph_final = np.array(temp_knowledge_graph_final)
        np.save(f, temp_knowledge_graph_final)
    with open(pheme_temporal_path+'label.npy','wb')as f:
        label_final = np.array(label_final)
        np.save(f, label_final)
    with open(pheme_temporal_path+'propagation_root_index.npy','wb')as f:
        root_index_final = np.array(root_index_final)
        np.save(f, root_index_final)

if __name__ == '__main__':
    node2index()
    main()
