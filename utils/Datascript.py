from __future__ import division

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0].min() <= 1e-5:
        A = A + torch.diag_embed(torch.ones(A.size(1), dtype=torch.float32).unsqueeze(0).expand(A.size(0),
                                                                                                -1)).to(
            A.device)  # if the diag has been added by 1s
    D = torch.sum(A, dim=1)
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = torch.reciprocal(torch.sqrt(D))
    # A_wave = torch.matmul(torch.matmul(diag.unsqueeze(1), A), diag.unsqueeze(2))
    A_wave = diag.unsqueeze(0) * A * diag.unsqueeze(1)
    return A_wave


def get_normalized_adj_npy(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    d = torch.sum(adj_mx, dim=1)
    d_inv = torch.pow(d, -1)
    d_inv[d_inv == float('inf')] = 0.
    d_mat_inv = torch.diag_embed(d_inv)
    random_walk_mx = torch.matmul(d_mat_inv, adj_mx)
    return random_walk_mx


def calculate_random_walk_matrix_npy(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    np.seterr(divide='ignore', invalid='ignore')
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def get_Aq_and_Ah(A):
    A_wave = get_normalized_adj_npy(A)
    A_q = calculate_random_walk_matrix_npy(A_wave.transpose(0, 1)).transpose(0, 1)
    A_h = calculate_random_walk_matrix_npy(A_wave.transpose(0, 1)).transpose(0, 1)
    return A_q, A_h


def prepare_time_info(datetime_str):
    # Example input: ["2023-10-01 08:00:00", ...]
    time_features = []
    for dt_str in datetime_str:
        dt = datetime.strptime(str(dt_str), "%Y-%m-%d %H:%M:%S")
        is_weekday = True if dt.weekday() < 5 else False
        is_morning_peak = True if 7 <= dt.hour < 9 else False
        is_evening_peak = True if 17 <= dt.hour < 19 else False
        time_features.append([is_weekday, is_morning_peak, is_evening_peak])

    return np.array(time_features).astype(np.bool_)


def fourier_time_embedding(timestamps: np.ndarray, num_freqs=4):
    """
    利用 Fourier embeddings 对 datetime 时间戳进行周期性编码
    :param timestamps: Datetime 数组
    :param num_freqs: 使用多少个频率
    """
    # 转换为秒数 (可用相对秒，也可用 Unix timestamp)
    t_seconds = np.array([t.astype('datetime64[s]').astype('int64') / 1e9 for t in timestamps])  # 转秒
    
    # 为避免数值过大，可以取模（一天内的秒数）
    seconds_in_day = 24 * 3600
    t_norm = (t_seconds % seconds_in_day) / seconds_in_day  # [0,1] 周期化
    
    embeddings = []
    for k in range(num_freqs):
        freq = 2.0 ** k  # 指数增长的频率
        embeddings.append(np.sin(2 * np.pi * freq * t_norm))
        embeddings.append(np.cos(2 * np.pi * freq * t_norm))
    return np.stack(embeddings, axis=-1)   # (N, num_freqs*2)



class MultiSourceTaxiData_long(Dataset):

    def __init__(self, X_List, num_timesteps_input, num_timesteps_output, provide_range: list, moe_activation_data=None):
        assert num_timesteps_input % 3 == 0, "num_timesteps_input must be a multiple of 3"
        assert isinstance(provide_range, list) and len(
            provide_range) == 2, "provide_range must be a list with 2 elements"
        assert isinstance(X_List, list), "X-List must be a python-list of numpy arrays"

        self.X_List = X_List
        self.num_sources = len(X_List)  # Number of input datasets (e.g., X1, X2, X3, ...)
        self.num_nodes = X_List[0].shape[0]
        self.moe_activation_data = moe_activation_data
        if moe_activation_data is not None:
            if isinstance(moe_activation_data, pd.DataFrame):
                # index变成列
                time_info = moe_activation_data.index.to_numpy()
                _v = moe_activation_data.values
                time_embedding = fourier_time_embedding(time_info)
                moe_activation_data = np.concatenate([time_embedding, _v], axis=1)
            else:
                moe_activation_data = prepare_time_info(moe_activation_data)
            moe_activation_data = np.expand_dims(moe_activation_data.transpose(1, 0), 1)
            self.X_List = [np.concatenate([X, moe_activation_data,], axis=0) for X in self.X_List]
            
        self.day_length = 72
        slot = num_timesteps_input // 3
        self.dataset_len = X_List[0].shape[2] - (7 * self.day_length + num_timesteps_output) + 1
        self.slot_length = 7 * self.day_length + slot
        day = self.day_length

        X_indices = [[lw for lw in range(i, i + slot)] +
                     [yd for yd in range(i + 6 * day, i + 6 * day + slot)] +
                     [td for td in range(i + 7 * day - slot, i + 7 * day)]
                     for i in range(self.dataset_len)]
        y_indices = [[_ for _ in range(i + 7 * day, i + 7 * day + num_timesteps_output)]
                     for i in range(self.dataset_len)]

        provide_floor = int(provide_range[0])
        provide_ceil = int(provide_range[1])

        X_indices = X_indices[provide_floor:provide_ceil]
        y_indices = y_indices[provide_floor:provide_ceil]

        self.features_list = [[] for _ in range(self.num_sources)]  # List of lists to store features for each source
        self.targets_list = [[] for _ in range(self.num_sources)]  # List of lists to store targets for each source

        for i, j in zip(X_indices, y_indices):
            for idx, X in enumerate(X_List):  # Loop through each dataset in X_List
                self.features_list[idx].append(self.X_List[idx][:, 0, i])
                self.targets_list[idx].append(X_List[idx][:, 0, j])

        # Convert lists to torch tensors
        self.features_list = [torch.from_numpy(np.array(features)) for features in self.features_list]
        self.targets_list = [torch.from_numpy(np.array(targets)) for targets in self.targets_list]

        # Update dataset length after slicing
        self.dataset_len = provide_range[1] - provide_range[0]
        assert self.dataset_len == len(y_indices), f"Dataset length mismatch: {self.dataset_len} vs {len(y_indices)}"
        pass
        # print("Multi-source Dataset Initialization Complete")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Retrieve features and targets for all datasets
        xList = [features[idx] for features in self.features_list]
        yList = [targets[idx] for targets in self.targets_list]
        xList = torch.stack(xList, dim=0)
        yList = torch.stack(yList, dim=0)

        return xList, yList # (mode, node_num, time_seq) or (batch, mode, node_num, time_seq)


class MultiSourceTaxiData_short(Dataset):
    def __init__(self, X_List, num_timesteps_input, num_timesteps_output, provide_range: list, num_days=30, moe_activation_data=None):
        assert num_timesteps_input % 3 == 0, "num_timesteps_input must be a multiple of 3"
        assert isinstance(provide_range, list) and len(
            provide_range) == 2, "provide_range must be a list with 2 elements"
        assert isinstance(X_List, list), "X-List must be a python-list of numpy arrays"

        self.X_List = X_List
        self.num_sources = len(X_List)  # Number of input datasets (e.g., X1, X2, X3, ...)

        self.dataset_len = X_List[0].shape[2] - num_timesteps_input - num_timesteps_output + 1

        X_indices = [[_ for _ in range(i, i + num_timesteps_input)] for i in range(self.dataset_len)]
        y_indices = [[_ for _ in range(i + num_timesteps_input, i + num_timesteps_input + num_timesteps_output)]
                     for i in range(self.dataset_len)]
        provide_floor = int(provide_range[0])
        provide_ceil = int(provide_range[1])

        X_indices = X_indices[provide_floor:provide_ceil]
        y_indices = y_indices[provide_floor:provide_ceil]

        self.features_list = [[] for _ in range(self.num_sources)]  # List of lists to store features for each source
        self.targets_list = [[] for _ in range(self.num_sources)]  # List of lists to store targets for each source

        for i, j in zip(X_indices, y_indices):
            for idx, X in enumerate(X_List):  # Loop through each dataset in X_List
                self.features_list[idx].append(X[:, :, i].transpose((0, 2, 1)))
                self.targets_list[idx].append(X[:, 0, j])

        # Convert lists to torch tensors
        self.features_list = [torch.from_numpy(np.array(features)) for features in self.features_list]
        self.targets_list = [torch.from_numpy(np.array(targets)) for targets in self.targets_list]

        # Update dataset length after slicing
        self.dataset_len = provide_range[1] - provide_range[0]

        # print("Multi-source Dataset Initialization Complete")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Retrieve features and targets for all datasets
        xList = [features[idx] for features in self.features_list]
        yList = [targets[idx] for targets in self.targets_list]

        return xList, yList


def make_MultiSourceTaxiData_long_loader(X_List,
                                         num_timesteps_input,
                                         num_timesteps_output,
                                         batch_size,
                                         split_ratio1=0.7,
                                         split_ratio2=0.8,
                                         # num_days=30,
                                         loadmode='',
                                         return_with_scaler=False,
                                         moe_avtivation_data=None):
    assert isinstance(X_List, list), "X-List must be a python-list of numpy arrays"

    # X_List = [X[:2160, :].T for X in X_List]
    X_List = [X.T for X in X_List]
    X_List = [X.astype(np.float32) for X in X_List]
    X_List = [X.reshape((X.shape[0], 1, X.shape[1])) for X in X_List]

    scalar_list = []

    for i, x in enumerate(X_List):
        X_List[i][X_List[i] > np.percentile(X_List[i], 99.9)] = np.percentile(X_List[i], 99.9)
       
    day_length = 72

    if loadmode == 'BQN':
        assert num_timesteps_input % 3 == 0, "num_timesteps_input must be a multiple of 3"
        dataset_len = X_List[0].shape[2] - (7 * day_length + num_timesteps_input // 3 + num_timesteps_output) + 1
        Ds = MultiSourceTaxiData_long
    else:
        dataset_len = X_List[0].shape[2] - num_timesteps_input - num_timesteps_output + 1
        Ds = MultiSourceTaxiData_short


    split_line1 = int(dataset_len * split_ratio1)
    split_line2 = int(dataset_len * split_ratio2)

    trainset = Ds(X_List,
                  num_timesteps_input=num_timesteps_input,
                  num_timesteps_output=num_timesteps_output,
                  provide_range=[0, split_line1],
                  moe_activation_data=moe_avtivation_data)
    testset = Ds(X_List,
                 num_timesteps_input=num_timesteps_input,
                 num_timesteps_output=num_timesteps_output,
                 provide_range=[split_line2, dataset_len],
                 moe_activation_data=moe_avtivation_data)

    # Robustness check
    if split_line1 < split_line2:
        valset = Ds(X_List,
                    num_timesteps_input=num_timesteps_input,
                    num_timesteps_output=num_timesteps_output,
                    provide_range=[split_line1, split_line2],
                    moe_activation_data=moe_avtivation_data)
    else:
        import warnings
        warnings.warn("Validation set is empty, Using test set instead.")
        valset = testset

    if return_with_scaler:
        ((DataLoader(trainset, batch_size=batch_size, shuffle=True), \
          DataLoader(valset, batch_size=batch_size, shuffle=False), \
          DataLoader(testset, batch_size=batch_size, shuffle=False),),
         scalar_list
         )
    else:
        return DataLoader(trainset, batch_size=batch_size, shuffle=True), \
            DataLoader(valset, batch_size=batch_size, shuffle=False), \
            DataLoader(testset, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    minute = '15'
    X1 = pd.read_parquet('../Datasets/Beijing/rs_dataset_df_' + minute + 'min.parquet').values
    X2 = pd.read_parquet('../Datasets/Beijing/kc_dataset_df_' + minute + 'min.parquet').values
    X3 = pd.read_parquet('../Datasets/Beijing/zc_dataset_df_' + minute + 'min.parquet').values
    X_List = [X1, X2, X3]
    trainloader, valloader, testloader = (
        make_MultiSourceTaxiData_long_loader(X_List,
                                             num_timesteps_input=12,
                                             num_timesteps_output=3,
                                             batch_size=128,
                                             split_ratio1=0.7,
                                             split_ratio2=0.8))

    for x, y in trainloader:
        print(x[0].shape, y[0].shape)
        break
    for x, y in valloader:
        print(x[0].shape, y[0].shape)
        break
    for x, y in testloader:
        print(x[0].shape, y[0].shape)
        break
