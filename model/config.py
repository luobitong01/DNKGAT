from path_zh import *
from data import *

class dynamic_graph_Config():

    def __init__(self):

        # basic
        self.model_name = "DynamicKnowledgeGraphAttention"
        self.graph_embedding_dim = 128
        self.hidden_dim = 128
        self.n_class = 1
        self.report_step_num = 10
        self.dropout_rate = 0.5
        # self.learning_rate = 5e-5
        self.min_learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.patience = 2
        self.train = 0.7
        self.val = 0.1
        self.test = 0.2

        # task specific
        self.text_max_length = 30#120
        self.pad_idx = 6691
        self.basis_num = 2
        self.use_text = True
        self.k_hop = 1

        # train
        self.gpu_id = "0"

        # init
        self.init()


    def init(self):
        ''' additional configuration '''
        self.entity_concept_size = 96787
        self.token_size = 258466
        # extra adjacent matrix number
        self.add_adj_size = 1 
