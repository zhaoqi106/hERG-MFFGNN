import os
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch


class MolNet(InMemoryDataset):
    """
    分子网络数据集类,继承自PyG的InMemoryDataset
    用于处理和存储分子图数据
    """
    def __init__(self, root='dataset', dataset=None, xd=None, y=None, transform=None, pre_transform=None, smile_graph=None):
        """
        初始化函数
        Args:
            root (str): 数据保存路径
            dataset (str): 数据集名称
            xd (list): SMILES分子表示列表
            y (list): 标签列表
            transform: 数据转换函数
            pre_transform: 数据预处理函数
            smile_graph (dict): SMILES到图数据的映射字典
        """
        # root用于保存原始数据和预处理后的数据
        super(MolNet, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """返回原始文件名列表"""
        return ['raw_file']

    @property
    def processed_file_names(self):
        """返回处理后的文件名列表"""
        return [self.dataset + '_pyg.pt']

    def download(self):
        """下载数据到raw_dir目录(此处未实现)"""
        pass

    def _process(self):
        """创建processed目录"""
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y, smile_graph):
        """
        处理数据并保存为PyG格式
        Args:
            xd (list): SMILES分子表示列表
            y (list): 标签列表
            smile_graph (dict): SMILES到图数据的映射字典
        """
        assert (len(xd) == len(y)), "smiles and labels must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data ', data_len)
        for i in range(data_len):
            smiles = xd[i]

            if smiles is not None:
                # 获取标签
                labels = np.asarray([y[i]])
                # 获取图结构数据
                graph_data = smile_graph[smiles]['graph_data']
                leng, features, edge_index, edge_attr, adj_order_matrix, dis_order_matrix = (
                   graph_data[0], graph_data[1], graph_data[2], graph_data[3], graph_data[4], graph_data[5])
                # 获取分子指纹数据
                morgan_data = smile_graph[smiles]['morgan_fp']
                maccs_data = smile_graph[smiles]['maccs_fp']
                rdit_data = smile_graph[smiles]['rdit_fp']

                if len(edge_index) > 0:
                    # 打印调试信息
                    print("Features:", features)
                    print("Edge Index:", edge_index)
                    print("Edge Attr:", edge_attr)
                    print("Labels:", labels)

                    # 创建PyG数据对象
                    GCNData = DATA.Data(
                        x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0).contiguous(),
                        edge_attr=torch.Tensor(edge_attr), y=torch.FloatTensor(labels)
                    )
                    # 添加额外属性
                    GCNData.adj = adj_order_matrix  # 邻接矩阵
                    GCNData.dis = dis_order_matrix  # 距离矩阵
                    GCNData.morgan_fp = morgan_data  # morgan指纹
                    GCNData.maccs_fp = maccs_data  # maccs指纹
                    GCNData.rdit_fp = rdit_data  # rdit指纹

                    # 添加长度和SMILES信息
                    GCNData.leng = [leng] 
                    GCNData.smi = smiles
                    data_list.append(GCNData)

        # 应用数据过滤
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # 应用数据转换
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')   
        # 整理数据并保存
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])