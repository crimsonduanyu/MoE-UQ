import math
import sys
sys.path.append('..')
import copy
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp

_MAP = 'Manhattan'
class LaplacianFilter(nn.Module):
    def __init__(self, space_dim, rank, activation='softmax', learnable=True):
        super(LaplacianFilter, self).__init__()

        self.space_dim = space_dim
        self.rank = rank
        self.activation = activation

        # self.m1 = nn.Parameter(torch.ones(space_dim, rank))                   # GraphWaveNet method
        # self.m2 = nn.Parameter(torch.ones(rank, space_dim))
        # nn.init.kaiming_normal_(self.m1, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.m2, mode='fan_in', nonlinearity='relu')
        if space_dim == 2667:
            print('init with priorG')
            g = np.load('Datasets/Beijing/GraphAdj.npy')
        else:
            print('init with priorG')
            g = np.load('Datasets/Manhattan/GraphAdj.npy')
        u,sig,v = np.linalg.svd(g)
        u = u*sig

        u = u[:, rank:]                # 丢掉前 rank 列
        v = v[rank:, :]
        if learnable:
            self.m1 = nn.Parameter(torch.tensor(u).float())
            self.m2 = nn.Parameter(torch.tensor(v).float())
        else:
            self.register_buffer("m1", torch.tensor(u, dtype=torch.float32))
            self.register_buffer("m2", torch.tensor(v, dtype=torch.float32))

        del g, u, sig, v


    def forward(self):
        mat = torch.matmul(self.m1, self.m2)
        #mat = (mat + mat.T) / 2.
        if self.activation == 'softmax':
            return F.softmax(mat, dim=1)
        elif self.activation == 'sigmoid':
            return F.sigmoid(mat)
        else:
            raise ValueError('Activation function not supported.')


class B_TCN(nn.Module):
    """
    Neural network block that applies a bidirectional temporal convolution to each node of
    a graph.
    From Dingyi's code
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        """
        :param in_channels: Number of nodes in the graph.
        :param out_channels: Desired number of output features.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(B_TCN, self).__init__()
        # forward dirction temporal convolution
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_timesteps, num_features)
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        Xf = X.unsqueeze(1)  # (batch_size, 1, num_timesteps, num_nodes)

        inv_idx = torch.arange(Xf.size(2) - 1, -1, -1).long().to(
            device=self.device)
        Xb = Xf.index_select(2, inv_idx)  # inverse the direction of time

        Xf = Xf.permute(0, 3, 1, 2)
        Xb = Xb.permute(0, 3, 1, 2)  # (batch_size, num_nodes, 1, num_timesteps)
        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))  # +
        outf = tempf + self.conv3(Xf)
        outf = outf.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))  # +
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])

        rec = torch.zeros([batch_size, self.kernel_size - 1, self.out_channels]).to(
            device=self.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)  # (batch_size, num_timesteps, out_features)

        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(device=self.device)
        outb = outb.index_select(1, inv_idx)
        out = outf + outb
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(outf) + F.sigmoid(outb)
        return out



class ChebGraphConv(nn.Module):
    def __init__(self, orders, in_channels, out_channels, activation='None'):
        super(ChebGraphConv, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def _concat(x, x_):
        return torch.cat([x, x_.unsqueeze(0)], dim=0)

    def forward(self, X, adj):
        batch_size, num_node, input_size = X.shape
        x0 = X.permute(1, 2, 0).reshape(num_node, input_size * batch_size)
        x = x0.unsqueeze(0)
        for _ in range(1, self.orders + 1):
            x1 = torch.mm(adj, x0)
            x = self._concat(x, x1)
            x0 = x1

        x = x.reshape(self.num_matrices, num_node, input_size, batch_size).permute(3, 1, 2, 0)
        x = x.reshape(batch_size, num_node, input_size * self.num_matrices)
        x = torch.matmul(x, self.Theta1) + self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)
        return x



class MultiLayerGcn(nn.Module):
    def __init__(self,
                 space_dim=None,
                 hidden_dim_s=100,
                 rank_s=30,
                 num_timesteps_input=12,
                 n_layer=3,
                 prior_G=None,
                 learnable_G=True):
        super(MultiLayerGcn, self).__init__()
        self.gcns = nn.ModuleList([])
        self.gcns.append(ChebGraphConv(3, num_timesteps_input, hidden_dim_s, activation='relu'))
        for i in range(n_layer - 3):
            self.gcns.append(ChebGraphConv(3, hidden_dim_s, hidden_dim_s, activation='linear'))
        self.gcns.append(ChebGraphConv(2, hidden_dim_s, rank_s, activation='linear'))
        self.gcns.append(ChebGraphConv(2, rank_s, hidden_dim_s, activation='relu'))

        self.prior = False if prior_G is None else True
        if prior_G is None:
            print('Using learnable Laplacian filters.')
            self.laplacian = nn.ModuleList([])
            _laplacian_rank = int(math.ceil(math.sqrt(space_dim))/2)
            for i in range(n_layer):
                self.laplacian.append(LaplacianFilter(space_dim, _laplacian_rank, activation='softmax', learnable=learnable_G))
                _laplacian_rank *= 2
        else:
            print('Using prior Laplacian filters.')
            self.laplacian = prior_G

    def forward(self, x):
        for i in range(len(self.gcns)):
            if self.prior:
                x = self.gcns[i](x, self.laplacian)
            else:
                x = self.gcns[i](x, self.laplacian[i]())
        return x


class MultiLayerTcn(nn.Module):
    def __init__(self,
                 space_dim=None,
                 hidden_dim_t=24,
                 rank_t=6,
                 n_layer=3):
        super(MultiLayerTcn, self).__init__()
        assert space_dim is not None, 'space_dim must be specified, usually the number of nodes'
        self.l1 = B_TCN(space_dim, hidden_dim_t, kernel_size=3, activation='relu')
        self.l2 = B_TCN(hidden_dim_t, rank_t, kernel_size=3, activation='linear')
        self.l3 = B_TCN(rank_t, hidden_dim_t, kernel_size=3, activation='relu')

    def forward(self, x):
        x = x.permute(0, 2, 1)  # torch.Size([Batch_size, T_step, space_dim])
        x = self.l1(x)  # torch.Size([Batch_size, T_step, hidden_dim_t])
        x = self.l2(x)  # torch.Size([Batch_size, T_step, rank_t])
        x = self.l3(x)  # torch.Size([Batch_size, T_step, hidden_dim_t])
        return x




class SpatialTemporalBackbone(nn.Module):
    def __init__(self, space_dim, hidden_dim_t=48, hidden_dim_s=128, rank_t=12, rank_s=30,
                 T_in=12, T_out=3, prior_G=None, learnable_G=True):
        super(SpatialTemporalBackbone, self).__init__()
        self.space_dim = space_dim
        self.backbone1_Temporal = MultiLayerTcn(space_dim, hidden_dim_t, rank_t)
        # self.backbone1_Temporal = MultiLayerLstm(space_dim, hidden_dim_t, rank_t)
        self.backbone1_Spatial = MultiLayerGcn(space_dim, hidden_dim_s, rank_s, T_in, prior_G=prior_G,learnable_G=learnable_G)

        self.shortcut = nn.Sequential()

        self.t_mlp = nn.Sequential(
            Mlp(hidden_dim_t, math.ceil(space_dim*1.5), space_dim, drop=0.05),
        )

        self.s_mlp = nn.Sequential(
            Mlp(hidden_dim_s, hidden_dim_s * 2,T_in, drop=0.05),
        )

        self.st_mlp = nn.Sequential(
            Mlp(T_in, hidden_dim_t * 2, hidden_dim_t, drop=0.1),
            Mlp(hidden_dim_t, hidden_dim_t, hidden_dim_t, drop=0.1),
        )

    def forward(self, x):
        S = self.shortcut(x)
        T = self.shortcut(x)

        S = self.backbone1_Spatial(S)
        T = self.backbone1_Temporal(T)

        S = self.s_mlp(S)
        T = self.t_mlp(T)

        x = S * T.permute(0, 2, 1)
        return self.st_mlp(x)


class RMQR(nn.Module):
    def __init__(self, quantile_list, input_dim, hidden_dim, output_dim):
        super(RMQR, self).__init__()
        self.quantile_list = sorted(quantile_list)
        self.quantile_num = len(quantile_list)
        assert self.quantile_num % 2 == 1, 'Quantile number must be odd.'
        self.shortcut = nn.Sequential()

        self.median, self.lower, self.upper = self._create_head(input_dim, hidden_dim, output_dim)


    def _create_head(self, input_dim, hidden_dim, output_dim):
        half_len = self.quantile_num // 2
        self.upper = nn.ModuleList([Mlp(input_dim, hidden_dim, output_dim, drop=0.05) for _ in range(half_len)])
        self.lower = nn.ModuleList([Mlp(input_dim, hidden_dim, output_dim, drop=0.05) for _ in range(half_len)])
        self.median = Mlp(input_dim, hidden_dim, output_dim)

        return self.median, self.lower, self.upper

    def forward(self, x):
        median = self.median(x)
        lower_result, upper_result = [], []
        l0 = self.shortcut(median)
        u0 = self.shortcut(median)
        for i in range(self.quantile_num // 2):
            l_delta = self.lower[i](self.shortcut(x))
            u_delta = self.upper[i](self.shortcut(x))
            l0 = self.shortcut(l0) - F.relu(l_delta)
            u0 = self.shortcut(u0) + F.relu(u_delta)
            lower_result.append(l0)
            upper_result.append(u0)
        lower_result.reverse()
        return torch.stack(lower_result + [median] + upper_result, dim=1)





class Gate(nn.Module):
    def __init__(self, num_task, num_nodes, num_T, num_experts):
        super(Gate, self).__init__()
        self.reduce_t = nn.Linear(num_T, 1)
        self.reduce_n = nn.Linear(num_nodes, num_experts)

    def forward(self, x):
        x = F.gelu(self.reduce_t(x).squeeze(-1))
        x = F.relu(self.reduce_n(x).squeeze(-1))
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b e -> b 1 1 e')
        return x



class EnvAwareRouter(nn.Module):
    def __init__(self, contextual_dim=13, t_in=24, num_experts=8, activate_top_k=3,  hidden_dim=64,tau=1.0):
        """
        Env-Aware Router: 根据时间特征 + 环境上下文特征路由专家
        :param time_dim: 时间嵌入维度（Fourier embedding维数）
        :param context_dim: 上下文特征维度
        :param hidden_dim: 隐藏层维度
        :param num_experts: 专家数量
        :param activate_top_k: 每次激活专家个数 (k-hot)
        :param tau: Gumbel-Softmax 温度参数
        """
        super(EnvAwareRouter, self).__init__()
        self.num_experts = num_experts
        self.activate_top_k = activate_top_k
        self.tau = tau

        # Router 网络
        self.c_mlp = Mlp(contextual_dim, hidden_dim, num_experts, drop=0.2)
        self.t_mlp = Mlp(t_in, hidden_dim, 1, drop=0.2)

    def forward(self, contextual):
        """
        :param time_feat: (batch, time_dim) 时间嵌入
        :param context_feat: (batch, context_dim) 上下文特征
        :return: k-hot mask (batch, num_experts)，soft_probs (batch, num_experts)
        """
        contextual = self.t_mlp(contextual).squeeze(-1)  # (batch, 1)
        logits = self.c_mlp(contextual)

        # soft Gumbel-Softmax 概率 (可微)
        probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)

        # top-k mask
        topk_idx = torch.topk(probs, self.activate_top_k, dim=-1).indices  # (batch, k)
        mask = torch.zeros_like(probs).scatter_(1, topk_idx, 1.0)

        # Straight-Through Estimator：前向用 hard mask，反向梯度通过 probs
        mask_ste = mask + (probs - probs.detach())

        return mask_ste, probs



class ST_MoE_RMQRN(nn.Module): 
    def __init__(self, space_dim, contextual_dim, hidden_dim_t=64, hidden_dim_s=128, rank_t=32, rank_s=60,
                 T_in=12, T_out=3, task_num=3, quantile_list=None, prior_G=None, num_experts=6):
        super(ST_MoE_RMQRN, self).__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.space_dim = space_dim
        self.contextual_dim = contextual_dim
        self.task_num = task_num
        self.hidden_dim_t = hidden_dim_t

        # Define the single-task quantile networks (QNet) for each task
        self.specific_experts = nn.ModuleList([SpatialTemporalBackbone(space_dim, hidden_dim_t, hidden_dim_s, rank_t, rank_s,
                                                              T_in, T_out, prior_G) for _ in range(task_num)])
        # Define the shared experts
        self.shared_experts = nn.ModuleList([SpatialTemporalBackbone(space_dim, hidden_dim_t, hidden_dim_s, rank_t, rank_s,
                                                              T_in, T_out, prior_G) for _ in range(num_experts)])
        self.EnvAwareRouter = EnvAwareRouter(contextual_dim=contextual_dim, 
                                             num_experts=num_experts, 
                                             activate_top_k=3,
                                             hidden_dim=64)


        # Define a separate gating mechanism for each task
        self.gates = nn.ModuleList([Gate(task_num, space_dim, hidden_dim_t, num_experts) for _ in range(task_num)])
        self.towers = nn.ModuleList([RMQR(quantile_list, hidden_dim_t, hidden_dim_t * 2, T_out) for _ in range(task_num)])
        self.shortcut = nn.Sequential()

    def forward(self, _x):
        x, contextual = torch.split(_x.float(), [self.space_dim, self.contextual_dim], dim=2)
        b, m, n, t_in = x.shape

        contextual_flat = rearrange(contextual, 'b m c t -> (b m) c t')
        mask, probs = self.EnvAwareRouter(contextual_flat)

        num_experts = mask.shape[-1]
        mask = rearrange(mask, '(b m) e -> b m e', b=b, m=m)
        probs = rearrange(probs, '(b m) e -> b m e', b=b, m=m)
        x_flat = rearrange(x, 'b m n t -> (b m) n t')  # 合并 m
        expert_outputs_list = [0] * num_experts
        for expert_idx in range(num_experts):
            # 找出当前专家对应的样本索引
            sample_mask = mask[:, :, expert_idx].view(-1)               # (b*m,)
            selected_idx = torch.nonzero(sample_mask, as_tuple=True)[0]

            if len(selected_idx) == 0:
                expert_outputs_list[expert_idx] = x_flat.new_zeros((b, m, n, self.hidden_dim_t))  # 当前专家没有样本，输出全0
                continue  # 跳过当前专家

            # 取出输入，调用专家 forward
            input_to_expert = x_flat[selected_idx]                      # (N_selected, n, t_in)
            output_from_expert = self.shared_experts[expert_idx](input_to_expert)  # (N_selected, n, t_out)

            # 初始化输出容器
            expert_outputs_flat = x_flat.new_zeros((b * m, n, self.hidden_dim_t))

            # 将结果写回对应位置
            expert_outputs_flat[selected_idx] = output_from_expert

            expert_outputs_list[expert_idx] = rearrange(expert_outputs_flat, '(b m) n t -> b m n t', b=b, m=m)

        shared_expert_output = torch.stack(expert_outputs_list, dim=-1)  # (b, m, n, t_out, num_experts)
        task_results = [self.specific_experts[i](x[:, i, ...]) for i in range(self.task_num)]

        for i in range(self.task_num):
            support = self.gates[i](self.shortcut(task_results[i]))
            weighted_shared_expert_output = torch.sum(support * shared_expert_output[:,i,...], dim=-1)
            task_results[i] = self.towers[i](weighted_shared_expert_output+self.shortcut(task_results[i]))

        return torch.stack(task_results, dim=1)
    
