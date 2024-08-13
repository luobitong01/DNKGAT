from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

class RippleKnowledgeGraphAttention_nBatch(Module):
    def __init__(self, in_features, hidden_features, out_features,dropout,device, bias=True):
        super(RippleKnowledgeGraphAttention_nBatch, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features * 4, out_features)
        self.dropout = dropout
        self.device = device
        self.n_layers = 3
        self.bn_knowledge = nn.BatchNorm1d(in_features)
        self.ripple_net = RippleNet()

    def forward(self, x_knowledge, head_embedding, rel_embedding, tail_embedding):
        
        all_embeddings = [x_knowledge]

        for k in range(0, self.n_layers):
            
            side_embedding = self.ripple_net(x_knowledge, head_embedding, rel_embedding, tail_embedding)

            sum_embedding = x_knowledge + side_embedding 
            sum_embedding = F.leaky_relu(self.linear1(sum_embedding)) 

            bi_embedding = x_knowledge * side_embedding
            bi_embedding = F.leaky_relu(self.linear1(bi_embedding)) 

            x_knowledge = bi_embedding + sum_embedding 

            x_knowledge = F.dropout(x_knowledge, self.dropout, training=self.training) 

            x_norm_knowledge = self.bn_knowledge(x_knowledge) 

            all_embeddings += [x_norm_knowledge]

        all_embeddings = torch.cat(all_embeddings, dim=1) 
        all_embeddings = self.linear2(all_embeddings) 

        return all_embeddings 

class RippleNet(Module):
    def __init__(self):
        super(RippleNet, self).__init__()

    def forward(self, x_knowledge, head_embedding, rel_embedding, tail_embedding):
        device = x_knowledge.device  
        h_emb_list = torch.stack(head_embedding).to(device)  
        r_emb_list = torch.stack(rel_embedding).to(device)   
        t_emb_list = torch.stack(tail_embedding).to(device) 

        Rh = h_emb_list * r_emb_list
        
        v = x_knowledge.mean(dim=0).unsqueeze(1).to(device)  

        probs = torch.matmul(Rh, v).squeeze(1)
        probs_normalized = F.softmax(probs, dim=0)
        probs_expanded = probs_normalized.unsqueeze(1)

        o = torch.sum(t_emb_list * probs_expanded, dim=0)

        return o

class CrossCompressUnit1(nn.Module):
    def __init__(self, in_features, max_num_knowledge, max_num_propagation, device):
        super(CrossCompressUnit1, self).__init__()
        self.in_features = in_features
        self.device = device

        self.weight_vv = nn.Parameter(torch.Tensor(in_features, 1).to(device))
        self.weight_ev = nn.Parameter(torch.Tensor(in_features, 1).to(device))
        self.weight_ve = nn.Parameter(torch.Tensor(in_features, 1).to(device))
        self.weight_ee = nn.Parameter(torch.Tensor(in_features, 1).to(device))
        self.bias_v = nn.Parameter(torch.zeros(in_features, 1).to(device))
        self.bias_e = nn.Parameter(torch.zeros(in_features, 1).to(device))

        init.xavier_uniform_(self.weight_vv)
        init.xavier_uniform_(self.weight_ev)
        init.xavier_uniform_(self.weight_ve)
        init.xavier_uniform_(self.weight_ee)

    def forward(self, v, e, num_knowledge, num_propagation):  
        
        origin_v = v 
        origin_e = e 
        
        v = v.mean(dim=0, keepdim=True).transpose(0, 1)  
        e = e.mean(dim=0, keepdim=True).transpose(0, 1)  
        
        v_transpose = v.transpose(0, 1) 
        e_transpose = e.transpose(0, 1) 

        c_matrix = torch.matmul(v, e_transpose) 
        c_matrix_transpose = torch.matmul(e, v_transpose)
        
        v_output = torch.matmul(c_matrix, self.weight_vv) + torch.matmul(c_matrix_transpose, self.weight_ev) 

        v_output = v_output + self.bias_v  

        e_output = torch.matmul(c_matrix, self.weight_ve) + torch.matmul(c_matrix_transpose, self.weight_ee) 
        e_output = e_output + self.bias_e 
        
        v_output = v_output.transpose(0, 1).expand_as(origin_v) 
        e_output = e_output.transpose(0, 1).expand_as(origin_e) 
        
        origin_v = origin_v + v_output
        origin_e = origin_e + e_output
        
        return origin_v, origin_e

class Mean_nBatch_geometric(Module):
    def __init__(self,device,n_feature,n_output):
        super(Mean_nBatch_geometric, self).__init__()
        self.device = device
        self.linear = nn.Linear(n_feature, n_output)
    def forward(self, x, batch):
        x = x.to(self.device)
        batch = batch.to(self.device)
        x = scatter_mean(x, batch, dim=0)
        return x
