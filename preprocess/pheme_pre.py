from urllib.request import quote
import requests
import os
import csv
import json
import pandas as pd
from tqdm import tqdm

pheme_raw_path = '../data/pheme/all-rnr-annotated-threads/'
pheme_clean_path = '../data/pheme/pheme_clean/'
pheme_entity_path = '../data/pheme/pheme_entity/'
pheme_clean_path_unfinish = '../data/pheme/pheme_clean_unfinish/'

def read_raw_data():
    news_ids = []
    label_dic = {}
    event_list = ['germanwings-crash-all-rnr-threads', 'charliehebdo-all-rnr-threads',
                  'sydneysiege-all-rnr-threads', 'ebola-essien-all-rnr-threads',
                  'gurlitt-all-rnr-threads', 'putinmissing-all-rnr-threads',
                  'ferguson-all-rnr-threads', 'ottawashooting-all-rnr-threads',
                  'prince-toronto-all-rnr-threads']
    for event in event_list:
        if event != '._.DS_Store':
            temp_nonrumor_path = pheme_raw_path+event+'/non-rumours/'
            temp_rumor_path = pheme_raw_path+event+'/rumours/'
            for news in os.listdir(temp_nonrumor_path):
                if news[0:2] != '._' and news != '.DS_Store':
                    news_ids.append(news)
                    label_dic[news] = '0'
                    news_df = pd.DataFrame(columns=['mid', 'parent', 'text', 't'])
                    with open(temp_nonrumor_path + news+'/source-tweets/'+news+'.json', 'r', errors="ignore", encoding="utf-8") as load_news:
                        file = json.load(load_news)
                        mid = str(file["id"])+'\t'
                        parent = str(file["in_reply_to_status_id"])+'\t'
                        text = ' '.join(file["text"].split())+'\t'
                        t = file["created_at"]+'\t'
                        news_df.loc[0] = [mid, parent, text, t]
                    comment_list = os.listdir(temp_nonrumor_path + news+'/reactions/')
                    for i, comment in enumerate(comment_list):
                        if comment[0:2] != '._' and comment != '.DS_Store':
                            print("comment",comment)
                            with open(temp_nonrumor_path + news + '/reactions/' + comment, 'r', errors="ignore",
                                      encoding="utf-8") as load_comment:
                                file_c = json.load(load_comment)
                                mid = str(file_c["id"]) + '\t'
                                parent = str(file_c["in_reply_to_status_id"]) + '\t'
                                text = file_c["text"] + '\t'
                                t = file_c["created_at"] + '\t'
                                news_df.loc[i+1] = [mid, parent, text, t]
                    pheme_output = pheme_clean_path + news + '.csv'
                    news_df.to_csv(pheme_output)

            for news in os.listdir(temp_rumor_path):
                if news[0:2] != '._' and news != '.DS_Store':
                    news_ids.append(news)
                    label_dic[news] = '1'
                    news_df = pd.DataFrame(columns=['mid', 'parent', 'text', 't'])
                    with open(temp_rumor_path + news + '/source-tweets/' + news + '.json', 'r', errors="ignore",
                              encoding="utf-8") as load_news:
                        file = json.load(load_news)
                        mid = str(file["id"]) + '\t'
                        parent = str(file["in_reply_to_status_id"]) + '\t'
                        text = file["text"] + '\t'
                        t = file["created_at"] + '\t'
                        news_df.loc[0] = [mid, parent, text, t]
                    comment_list = os.listdir(temp_rumor_path + news + '/reactions/')
                    for i, comment in enumerate(comment_list):
                        if comment[0:2] != '._' and comment != '.DS_Store':
                            with open(temp_rumor_path + news + '/reactions/' + comment, 'r', errors="ignore",
                                      encoding="utf-8") as load_comment:
                                file = json.load(load_comment)
                                mid = str(file["id"]) + '\t'
                                parent = str(file["in_reply_to_status_id"]) + '\t'
                                text = file["text"] + '\t'
                                t = file["created_at"] + '\t'
                                news_df.loc[i + 1] = [mid, parent, text, t]
                    pheme_output = pheme_clean_path + news + '.csv'
                    news_df.to_csv(pheme_output)
    with open('../data/pheme/pheme_id_label.txt','w',encoding='utf-8',newline='')as f:
        for id in label_dic:
            string = id+'\t'+label_dic[id]+'\n'
            f.writelines(string)

def read_unfinish_raw_data():
    news_ids = []
    label_dic = {}
    event_list = ['germanwings-crash-all-rnr-threads', 'charliehebdo-all-rnr-threads',
                  'sydneysiege-all-rnr-threads', 'ebola-essien-all-rnr-threads',
                  'gurlitt-all-rnr-threads', 'putinmissing-all-rnr-threads',
                  'ferguson-all-rnr-threads', 'ottawashooting-all-rnr-threads',
                  'prince-toronto-all-rnr-threads']
    with open(pheme_senti_path+'unfinished_files.txt','r',encoding='utf-8')as f:
        unfinished_files = f.readlines()
    unfinished_file_ids = []
    for i in unfinished_files:
        print(i)
        file_id = i.split('\n').split('.')[0]
        unfinished_file_ids.append(file_id)
    print(unfinished_file_ids)

def entity_get(tweet):
    s = requests.session()
    url = 'https://tagme.d4science.org/tagme/gui'
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept-Encoding": "gzip,deflate,br",
        # "User-Agent":"python-requests/2.22.0",
        "Accept-Language": "zh-CN,zh;q=0.8",
        "Connection": "keep-alive",
        # "Cookie":"JSESSIONID=D4E5CD25398932886BCA4537AB99F910"
    }
    cookie = {
        'JSESSIONID': '2C38096BA406DE03424810EBD391094F'
    }
    count = 0
    text = tweet
    params = {'lang': 'en', 'text': text, 'rho': '50'}
    r = s.post(url, data=params, headers=headers, cookies=cookie)
    dict_text = r.json()
    sub_entity = []
    for anno in dict_text['annotations']:
        if 'title' in anno:
            sub_entity.append(anno['title'])
    print(sub_entity)
    return sub_entity

def read_text():
    pheme_files = os.listdir(pheme_clean_path)
    pheme_files_num = len(pheme_files)
    pheme_finished_files = os.listdir(pheme_entity_path)
    for i in tqdm(range(pheme_files_num)):
        file_name = pheme_files[i].split('.')[0]
        if pheme_files[i] not in pheme_finished_files:
            file_df = pd.read_csv(pheme_clean_path + pheme_files[i])
            file_df_new = pd.DataFrame(columns=['mid', 'parent', 'text', 't', 'entity'])
            entity_all = []
            for j in range(len(file_df['text'])):
                try:
                    mid = str(file_df['mid'][j]).strip('\t') + '\t'
                    parent = file_df['parent'][j].strip('\t') + '\t'
                    text = file_df['text'][j].strip('\t') + '\t'
                    t = str(file_df['t'][j]).strip('\t') + '\t'
                except AttributeError:
                    with open(pheme_entity_path+'unfinished_files.txt','a',encoding='utf-8',newline='')as f:
                        string = pheme_files[i] + '\n'
                        f.writelines(string)
                        break
                else:
                    entity_list = str(entity_get(text.strip('\t'))) + '\t'
                    file_df_new.loc[j] = [mid, parent, text, t, entity_list]
            pheme_output = pheme_entity_path + file_name + '.csv'
            file_df_new.to_csv(pheme_output)

read_raw_data()
read_text()
