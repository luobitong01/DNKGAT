from urllib.request import quote
import requests
import json
import csv
import os
import pandas as pd
from tqdm import tqdm

weibo_path = '../data/weibo/Weibo_raw/'
weibo_output_path = '../data/weibo/weibo_clean/'
weibo_entity_path = '../data/weibo/weibo_entity/'

def read_json():
    weibo_files = os.listdir(weibo_path)
    weibo_files_num = len(weibo_files)

    for i in range(weibo_files_num):
        file_name = weibo_files[i].split('.')[0]
        file_df = pd.DataFrame(columns=['mid','parent','text','t'])
        with open(weibo_path+weibo_files[i],'r',errors="ignore", encoding="utf-8") as load_news:
            file = json.load(load_news)
            # file_df.loc[num]
            print(type(file))#list
            file_len = len(file)
        for j in range(file_len):
            mid = file[j]['mid']+'\t'
            parent = str(file[j]['parent'])+'\t'
            text = file[j]['original_text']+'\t'
            t = str(file[j]['t'])+'\t'
            file_df.loc[j] = [mid,parent,text,t]
        weibo_output = weibo_output_path+file_name+'.csv'
        file_df.to_csv(weibo_output)

def read_text():
    weibo_files = os.listdir(weibo_output_path)
    weibo_files_num = len(weibo_files)
    for i in tqdm(range(weibo_files_num)[0:1]):
        file_name = weibo_files[i].split('.')[0]
        file_df = pd.read_csv(weibo_output_path+weibo_files[i])
        file_df_new = pd.DataFrame(columns=['mid','parent','text','t','entity'])
        # print(file_df['text'][0])
        entity_all = []
        for j in range(len(file_df['text'])):
            mid = str(file_df['mid'][j]).strip('\t')+'\t'
            parent = file_df['parent'][j].strip('\t')+'\t'
            text = file_df['text'][j].strip('\t')+'\t'
            t = str(file_df['t'][j]).strip('\t')+'\t'
            entity_list =  str(entity_get(text.strip('\t')))+'\t'
            file_df_new.loc[j] = [mid, parent, text, t,entity_list]
        weibo_output = weibo_entity_path+file_name+'.csv'
        file_df_new.to_csv(weibo_output)

def read_unfinished_text():
    weibo_files = os.listdir(weibo_output_path)
    weibo_files_num = len(weibo_files)
    weibo_entity_files = os.listdir(weibo_entity_path)
    weibo_entity_files_num = len(weibo_entity_files)
    count = 0
    for file in weibo_files:
        if file not in weibo_entity_files:
            print(file)
            count += 1
            file_name = file.split('.')[0]
            file_df = pd.read_csv(weibo_output_path + file)
            file_df_new = pd.DataFrame(columns=['mid', 'parent', 'text', 't', 'entity'])
            # print(file_df['text'][0])
            entity_all = []
            for j in range(len(file_df['text'])):
                mid = str(file_df['mid'][j]).strip('\t') + '\t'
                parent = file_df['parent'][j].strip('\t') + '\t'
                text = file_df['text'][j].strip('\t') + '\t'
                t = str(file_df['t'][j]).strip('\t') + '\t'
                entity_list = str(entity_get(text.strip('\t'))) + '\t'
                file_df_new.loc[j] = [mid, parent, text, t, entity_list]
                # entity_all.append(entity_list)
            weibo_output = weibo_entity_path + file_name + '.csv'
            file_df_new.to_csv(weibo_output)
    print(count)

def entity_get(text):
    urlEncode = quote(text)
    url = 'http://shuyantech.com/entitylinkingapi?q='+urlEncode+'&apikey=2aeae65c1378851a7a290703f4285612'
    headers = {
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accepting-Encoding':'gzip, deflate, sdch',
        'Content-Type':'text/html; charset=utf-8',
        'Accept - Language': 'zh-CN,zh;q=0.8',
        'Cache-Control':'max-age=0',
        'Proxy-Connection': 'keep-alive',
        'Host':'shuyantech.com',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'
        }
    params = {'q':text,'apikey':'2aeae65c1378851a7a290703f4285612'}
    res = requests.get(url,headers = headers,data = params)
    entity_list = []
    for en in res.json()['entities']:
        entity_list.append(en[1])
    print(entity_list)
    return entity_list

def read_file_label():
    weibo_files = os.listdir(weibo_path)
    weibo_files_num = len(weibo_files)
    with open('../data/weibo/weibo_id_label.txt', 'w', encoding='utf-8', newline='')as f:
        for i in range(weibo_files_num):
            file_name = weibo_files[i].split('.')[0]
            with open(weibo_path + weibo_files[i], 'r', errors="ignore", encoding="utf-8") as load_news:
                file = json.load(load_news)
                print(type(file))  
                file_len = len(file)
            label = str(file[0]['verified'])

            string = str(file_name)+'\t'+label+'\n'
            f.writelines(string)

read_json()
read_text() 
# read_unfinished_text()
read_file_label()


