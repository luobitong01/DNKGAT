#data-flow p_node -->GRU--> GCN-->fusioned
#          k_node       -->GCN -->fusioned
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
# from torch_sparse import spmm
from layers import *
from tqdm import tqdm
from transformers import BertModel, BertConfig

class DynamicKnowledgeGraphAttention(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=2, dropout=0.2, instance_norm=False):
        super(DynamicKnowledgeGraphAttention, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
#         self.mid2time = mid2time  
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
#         self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)

        self.n_feature = self.config.hidden_dim

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)
        
        #------------------------RippleKGAT--------------------------
        self.Ripple_KnowledgeGraphAttention_stack_knowledge = nn.ModuleList()
        for i in range(2):
            self.Ripple_KnowledgeGraphAttention_stack_knowledge.append(RippleKnowledgeGraphAttention_nBatch(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            
        #------------------------MKR--------------------------
        self.max_num_knowledge = len(self.token_dict)
        self.max_num_propagation = len(self.mid2bert_tokenizer)
        self.MKR_stack_knowledge_propagation = CrossCompressUnit1(self.n_feature, self.max_num_knowledge, self.max_num_propagation, self.device)
        
        #------------------------Attention--------------------------
#         self.geometric_attention = TemporalEncoding_nBatch_geometric_attention(self.n_feature, self.n_feature)
        
        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        self.fc_rumor_0 = nn.Linear(256,128)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        
        features_propagation = []
        features_knowledge = []
#         features_text_propagation = []
        
        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        
        len_propagation = len(idx_propagation_list)
        len_knowledge = len(idx_knowledge_list)
        
        # 根节点文本嵌入
        root_idx_p = []
        root_idx_k = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0) # 最终：32, 128
            root_idx_p.append(idx_propagation_list.index(root_idx_item))
            root_idx_k.append(idx_knowledge_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        root_idx_k = torch.tensor(root_idx_k)
        
#         print("text_encoding_bert : ", text_encoding_bert.size())
        # 传播节点嵌入
        for idx in data.x:
#         for idx in data.x_propagation_idx:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            features_propagation.append(text_embedding.detach().cpu().numpy()) # len_propagation, 128
        
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
#         for idx in data.x:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy()) # len_knowledge, 128
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx)) 
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx)) 

        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)
#         features_text_propagation = torch.tensor(features_text_propagation)
        
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)
            
        x_propagation = features_propagation # 399, 1, 128
        x_knowledge = features_knowledge # 13694, 1, 128
           
        x_propagation = torch.squeeze(x_propagation,dim=1) # 339, 128
        x_knowledge = torch.squeeze(x_knowledge,dim=1) # 13694, 128

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        
        def edge_indices_to_sparse_matrix(edge_indices, edge_values, size):
            return torch.sparse_coo_tensor(edge_indices, edge_values, torch.Size(size))

        def tensor_to_set(tensor):
            """
            将二维张量转换为集合，假设每一列是一个唯一的边。
            """
            return set(map(tuple, tensor.t().tolist()))
        
        def find_new_edges_and_values(old_set, new_set, new_indices, new_values):
            """
            找出新集合中相对于旧集合多出来的边及其对应的值。
            :param old_set: 旧时刻边的集合。
            :param new_set: 新时刻边的集合。
            :param new_indices: 新时刻边的索引。
            :param new_values: 新时刻边的值。
            :return: 多出来的边及其对应的值。
            """
            # 找出新集合中相对于旧集合多出来的边
            new_edges = new_set - old_set
            # 初始化一个空列表来存储多出来的边对应的值
            new_edge_values = []
            # 遍历新时刻的边的索引
            for i, edge in enumerate(new_indices.t().tolist()):
                if tuple(edge) in new_edges:
                    # 如果这条边是新出现的，添加其对应的值到列表中
                    new_edge_values.append(new_values[i])
            return list(new_edges), new_edge_values
        
        def find_cur_edges_and_values(set, new_indices, new_values):
            """
            找出当前集合的边及其对应的值。
            :param set: 当前时刻边的集合。
            :param new_indices: 当前时刻边的索引。
            :param new_values: 当前时刻边的值。
            :return: 当前时刻的边及其对应的值。
            """
            # 初始化一个空列表来存储当前时刻的边对应的值
            new_edge_values = []
            # 遍历当前时刻的边的索引
            for i, edge in enumerate(new_indices.t().tolist()):
                if tuple(edge) in set:
                    new_edge_values.append(new_values[i])
            return list(set), new_edge_values
        
        def random_sample_edges_and_values(edges, values, n):
            """
            从边和对应的值中随机抽取n个元素。
            :param edges: 边的列表。
            :param values: 边对应的值的列表。
            :param n: 要抽取的元素数量。
            :return: 随机抽取的边和对应的值。
            """
            n = min(len(edges), n)
#             if n > len(edges):
#                 raise ValueError("n cannot be greater than the number of edges")
            # 生成随机索引
            indices = random.sample(range(len(edges)), n)
            # 根据索引抽取元素
            sampled_edges = [edges[i] for i in indices]
            sampled_values = [values[i] for i in indices]
            return sampled_edges, sampled_values
        
        set_indices_0 = tensor_to_set(data.edge_index)
        set_indices_1 = tensor_to_set(data.x_knowledge_node_indices_1_edge_index)
        set_indices_2 = tensor_to_set(data.x_knowledge_node_indices_2_edge_index)
        
        new_edges_2_to_1, new_edge_values_2_to_1 = find_new_edges_and_values(set_indices_1, set_indices_2, data.x_knowledge_node_indices_2_edge_index, data.x_knowledge_node_values_2)
        new_edges_1_to_0, new_edge_values_1_to_0 = find_new_edges_and_values(set_indices_0, set_indices_1, data.x_knowledge_node_indices_1_edge_index, data.x_knowledge_node_values_1)
        
        head_embedding = []
        rel_embedding = []
        tail_embedding = []

        head_1_embedding = []
        rel_1_embedding = []
        tail_1_embedding = []
        
        head_2_embedding = []
        rel_2_embedding = []
        tail_2_embedding = []
        # 假设每次随机抽取的边的数量
        num_samples = 64

        sampled_edges_1_to_0, sampled_values_1_to_0 = random_sample_edges_and_values(new_edges_1_to_0, new_edge_values_1_to_0, num_samples)
        for i, (head_idx, tail_idx) in enumerate(sampled_edges_1_to_0):
            head_1_embedding.append(x_knowledge[head_idx])  # 头节点嵌入
            tail_1_embedding.append(x_knowledge[tail_idx])  # 尾节点嵌入
            rel_1_embedding.append(torch.FloatTensor([sampled_values_1_to_0[i]] * self.n_feature)) # 关系嵌入
        
        head_embedding.append(head_1_embedding)
        rel_embedding.append(rel_1_embedding)
        tail_embedding.append(rel_1_embedding)
        
        # 对 new_edges_2_to_1 进行随机抽样
        sampled_edges_2_to_1, sampled_values_2_to_1 = random_sample_edges_and_values(new_edges_2_to_1, new_edge_values_2_to_1, num_samples)
        for i, (head_idx, tail_idx) in enumerate(sampled_edges_2_to_1):
            head_2_embedding.append(x_knowledge[head_idx])  # 头节点嵌入
            tail_2_embedding.append(x_knowledge[tail_idx])  # 尾节点嵌入
            rel_2_embedding.append(torch.FloatTensor([sampled_values_2_to_1[i]] * self.n_feature)) # 关系嵌入
        
        head_embedding.append(head_2_embedding)
        rel_embedding.append(rel_2_embedding)
        tail_embedding.append(rel_2_embedding)
        for i, layer in enumerate(self.Ripple_KnowledgeGraphAttention_stack_knowledge):
            if len(head_embedding[i]) == 0:
                print("--- models --- zero new edges")
                continue
            x_knowledge = layer(x_knowledge, head_embedding[i], rel_embedding[i], tail_embedding[i])
        # 知识图和传播图进行MKR融合
        x_knowledge, x_propagation = self.MKR_stack_knowledge_propagation(x_knowledge, x_propagation, len_knowledge, len_propagation)
        x = self.mean(x_propagation, data.batch)
#         x = self.mean(x_knowledge, data.batch)
#         print(f"------------------x------------------:{x.size()}")
#         print(f"x 的设备: {x.device}")
        x = x.squeeze(1) # 32, 128
#         print(f"------------------x------------------:{x.size()}")
        x_fusion = torch.cat([x,text_encoding_bert],dim=1) # 32, 256
#         print(f"------------------x_fusion------------------:{x_fusion.size()}")
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion)) # 32, 100
#         print(f"------------------x_fusion------------------:{x_fusion.size()}")
        x_fusion = self.fc_rumor_2(x_fusion) # 32, 2
#         print(f"------------------x_fusion------------------:{x_fusion.size()}")
        x_fusion = torch.sigmoid(x_fusion) # 32, 2
#         print(f"------------------x_fusion------------------:{x_fusion.size()}")
        return x_fusion
