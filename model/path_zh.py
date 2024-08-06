import os
import sys


class path_Set_BERT():
    def __init__(self,dataset, dataset_dir=""):
        self.path_data_dir = os.path.join(dataset_dir, "data/{}".format(dataset))

        # vocab
        # self.path_entity_concept_2id = os.path.join(self.path_data_dir, "entity_concept_2id.csv")
        # path_relation2id = os.path.join(path_data_dir, "relation2id.txt")
        # self.path_token2id = os.path.join(self.path_data_dir, "token2id.csv")

        #mapped files
        # self.path_mid2token = os.path.join(self.path_data_dir, "mid2token.csv")
        self.path_node2idx_mid = os.path.join(self.path_data_dir, "node2idx_mid.txt")
        self.path_mid2text = os.path.join(self.path_data_dir, 'mid2text.txt')

        # data
        self.path_temporal = os.path.join(self.path_data_dir, "{}_temporal_data".format(dataset))
        self.path_temporal_propagation_idx = os.path.join(self.path_data_dir, "{}_temporal_data/propagation_node_idx.npy".format(dataset))
        self.path_temporal_knowledge_idx = os.path.join(self.path_data_dir, "{}_temporal_data/knowledge_node_idx.npy".format(dataset))
        self.path_temporal_propagation_graph = os.path.join(self.path_data_dir, "{}_temporal_data/propagation_node.npy".format(dataset))
        self.path_temporal_knowledge_idx = os.path.join(self.path_data_dir, "{}_temporal_data/knowledge_node.npy".format(dataset))
        self.path_label = os.path.join(self.path_data_dir, "{}_temporal_data/label.npy".format(dataset))
        self.path_root_idx = os.path.join(self.path_data_dir, "{}_temporal_data/propagation_root_index.npy".format(dataset))

        #trained_model
        self.path_saved_model = "model_saved/"
        #BERT_PATH
        if dataset == 'weibo':
            self.path_bert = '../bert-base-chinese/'
            self.VOCAB = 'vocab.txt'
        elif dataset == 'pheme':
            self.path_bert = '../bert-base-uncased/'
            self.VOCAB = 'vocab.txt'
