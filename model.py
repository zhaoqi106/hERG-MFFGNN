import torch.nn as nn
import torch
from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from co_attention import ParallelCoAttentionNetwork
from rdkit import Chem
from rdkit.Chem import Descriptors
from embed import Fox
from utils import create_ffn
import torch.nn.utils.rnn as rnn

# 计算分子描述符的函数
def calculate_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = list([
            Descriptors.MolWt(mol),  # 分子量
            Descriptors.MolLogP(mol),  # LogP
            Descriptors.TPSA(mol),  # 极性表面面积
            Descriptors.HeavyAtomCount(mol),  # 重原子计数
            Descriptors.NumRotatableBonds(mol)  # 可旋转键计数
        ])
        return descriptors
    else:
        return None

def fc_set(dim, emb_dim, dropout):
    return nn.Sequential(
            nn.Linear(dim, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, emb_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
        )


class Separated_MolFPEncoder(nn.Module):
    def __init__(self, emb_dim=128, fp_type=None, dropout=0.1, device='cuda:0'):
        super(Separated_MolFPEncoder, self).__init__()
        self.fp_type = fp_type
        self.device = device
        morgan_dim = 2048 if 'morgan' in fp_type else 0
        maccs_dim = 167 if 'maccs' in fp_type else 0
        rdit_dim = 2048 if 'rdit' in fp_type else 0

        self.des_fc = nn.Sequential(
            nn.Linear(5, 1)
        )
        self.batch_norm = nn.BatchNorm1d(emb_dim)
        self.act_func = nn.ReLU()
        self.morgan_fc = fc_set(morgan_dim+1, emb_dim, dropout)
        self.maccs_fc = fc_set(maccs_dim+1, emb_dim, dropout)
        self.rdit_fc = fc_set(rdit_dim+1, emb_dim, dropout)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        morgan_fp = data.morgan_fp.to(self.device) if 'morgan' in self.fp_type else torch.empty(0)
        maccs_fp = data.maccs_fp.to(self.device) if 'maccs' in self.fp_type else torch.empty(0)
        rdit_fp = data.rdit_fp.to(self.device) if 'rdit' in self.fp_type else torch.empty(0)

        descriptor_values_list = []
        for smiles in data.smi:
            if smiles is not None:
                # 计算分子描述符
                descriptors = calculate_molecular_descriptors(smiles)
                if descriptors is not None:
                    descriptor_values = descriptors
                else:
                    descriptor_values = list([0])*5  # 如果没有计算出描述符，则填充零
                descriptor_values_list.append(descriptor_values)
        descriptor_values_tensor = torch.tensor(descriptor_values_list, dtype=torch.float32, device=self.device)
        descriptor_feature = self.des_fc(descriptor_values_tensor)
        morgan_fp = self.morgan_fc(torch.cat([morgan_fp, descriptor_feature], dim=1)).unsqueeze(1)
        maccs_fp = self.maccs_fc(torch.cat([maccs_fp, descriptor_feature], dim=1)).unsqueeze(1)
        rdit_fp = self.rdit_fc(torch.cat([rdit_fp, descriptor_feature], dim=1)).unsqueeze(1)

        return torch.cat([morgan_fp, maccs_fp, rdit_fp], dim=1).permute(0,2,1)

class TaskSpecificOutput(nn.Module):  # 每个任务的单独输出
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(TaskSpecificOutput, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        output = self.fc(features)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output 
    
class Multitask_attn_fp(nn.Module):
    def __init__(self, dropout=0.1, device='cuda:0', fp_type=None, hidden_dim=256, num_heads=4):
        super(Multitask_attn_fp, self).__init__()
        # 使用 Fox 模型
        self.sep_enc = Separated_MolFPEncoder(emb_dim=hidden_dim, device=device, dropout = dropout, fp_type=fp_type)
        # self.enc = MolFPEncoder(emb_dim=hidden_dim, drop_ratio=dropout, device=device, fp_type=fp_type)
        # self.fp_fusion = TaskSpecificOutput(hidden_dim*2, hidden_dim*4, hidden_dim, dropout)
        # self.graph_fp1_fusion = TaskSpecificOutput(hidden_dim*2, hidden_dim*4, hidden_dim, dropout)
        # self.graph_fp2_fusion = TaskSpecificOutput(hidden_dim*2, hidden_dim*4, hidden_dim, dropout)
        self.unigram_conv = nn.Conv1d(hidden_dim, hidden_dim, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(hidden_dim, hidden_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=1, padding=2, dilation=2)
        self.max_pool = nn.MaxPool2d((3, 1))
        self.tanh = nn.Tanh()
        # self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=3, dropout=0.4)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.phrase_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.coattn = ParallelCoAttentionNetwork(hidden_dim, hidden_dim//4, src_length_masking=False)
        self.attn = Fox(attn_head=num_heads, output_dim= hidden_dim, device=device, attn_layers=8, dropout=dropout) 
        self.W_w = nn.Linear(hidden_dim, hidden_dim)
        self.W_p = nn.Linear(hidden_dim*2, hidden_dim)
        self.W_s = nn.Linear(hidden_dim*2, hidden_dim)
        self.th = nn.Tanh()
        self.sm = nn.Softmax(-1)
        # self.W_fp = nn.Linear(hidden_dim*2, hidden_dim)
        # self.norm = nn.BatchNorm1d(hidden_dim*2)
        # 每个任务的单独输出层
        self.output_layer = TaskSpecificOutput(hidden_dim, hidden_dim*2, 1, dropout)

    def forward(self, data): 

        # 通过 Fox 提取特征
        fp_feature1 = self.sep_enc(data).permute(2,0,1)
        # fp_feature2 = self.enc(data)
        graph_feature = self.attn(data).permute(0,2,1)
        # lens = torch.tensor([graph_feature.shape[2]] * graph_feature.shape[0])
        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(graph_feature)), 2) # B x 512 x L
        bigrams  = torch.unsqueeze(self.tanh(self.bigram_conv(graph_feature)), 2)  # B x 512 x L
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(graph_feature)), 2) # B x 512 x L
        graph_feature = graph_feature.permute(0,2,1)
        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(2, 0, 1) 
        # hidden = None
        # phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)
        # sentence_packed, hidden = self.lstm(phrase_packed, hidden)
        sentence_packed, _ = self.phrase_attn(phrase,phrase,phrase)
        sentence = sentence_packed.permute(1,0,2)
        phrase = phrase.permute(1,0,2)
        # sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        # sentence = torch.transpose(sentence, 0, 1)  # B x L x 512
        fp_feature1, _ = self.attention(fp_feature1, fp_feature1, fp_feature1)
        fp_feature1 = fp_feature1.permute(1,2,0)

        # Perform co-attention 
        
        _, _, v_word, q_word = self.coattn(fp_feature1, graph_feature, None)
        _, _, v_phrase, q_phrase = self.coattn(fp_feature1, phrase, None)
        _, _, v_sent, q_sent = self.coattn(fp_feature1, sentence, None)
 
        # _, _, fp_feature1, graph_feature = self.coattn(fp_feature1, graph_feature, None)

        # fp_fusion = self.fp_fusion(torch.cat([fp_feature1, fp_feature2], dim=1))
        # graph_fp1 = self.graph_fp1_fusion(torch.cat([graph_feature, fp_feature1], dim=1))
        # graph_fp2 = self.graph_fp2_fusion(torch.cat([graph_feature, fp_feature2], dim=1))

        # features_fusion = torch.cat((fp_fusion, graph_fp1, graph_fp2), dim=1)
        # features_fusion = self.norm(graph_fp1)
        # out = self.output_layer(features_fusion).sigmoid()
        h_w = self.tanh(self.W_w(q_word + v_word))
        # return h_w
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        # return h_p
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_p), dim=1)))

        # h_f = self.tanh(self.W_fp(torch.cat((fp_feature2, h_s), dim=1)))

        logit = self.output_layer(h_s).sigmoid()

        return logit

    

    
