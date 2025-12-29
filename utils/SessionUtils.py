import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append('..')
import math
import warnings
import random
from utils import Metrics
import json
import torch.nn.functional as F
import torch.optim as optim
import time

class SessionBase(object):
    def __init__(self, model:nn.Module, loss_fn, optimizer=None, device=None, early_stopping_patience=None, env_dict=None):
        self.now_epoch = 0
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        if early_stopping_patience is None:
            early_stopping_patience = 100
        else:
            assert early_stopping_patience >= 30, "Early stopping patience too small"
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.88)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min',factor=0.3, patience=15)
        self.loss_fn = loss_fn
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.trainlossList = []
        self.vallossList = []
        self.cityname = env_dict['cityname']
        self.time_granularity = env_dict['time_granularity']
        self.dataroot = env_dict['dataroot']
        self.pthroot = env_dict['pthroot']
        self.datapath = env_dict['datapath']
        self.pthpath = env_dict['pthpath']

    def deliver_batch(self, x_batch, y_batch):
        raise NotImplementedError

    def _train_a_epoch(self, train_loader):
        epoch_training_losses = []
        for x_batch_list, y_batch_list in train_loader:
            self.optimizer.zero_grad()
            loss, _ = self.deliver_batch(x_batch_list, y_batch_list)
            loss += 1e-4 * sum([torch.norm(param) for param in self.model.parameters()])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
            self.optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy().item())
        self.trainlossList.append(sum(epoch_training_losses) / len(epoch_training_losses))
        return self.trainlossList[-1]

    def _validate(self, val_loader):
        epoch_val_losses = []
        mae_list = []
        R2_list = []
        PICP_list = []
        with torch.no_grad():
            for x_batch_list, y_batch_list in val_loader:
                loss, (xb, yb, yh_) = self.deliver_batch(x_batch_list, y_batch_list)
                epoch_val_losses.append(loss.detach().cpu().numpy().item())
                yh = self.select_y_hat(yh_)
                mae_list.append(Metrics.mae_loss(yb, yh).detach().cpu().numpy().item())
                R2_list.append(Metrics.R_squared(yb.detach().cpu().numpy(), yh.detach().cpu().numpy()))
                picp = np.sum(
                    np.logical_and(
                    np.less_equal(yh_.detach().cpu().numpy()[:,:,0,...],yb.detach().cpu().numpy()),
                    np.greater_equal(yh_.detach().cpu().numpy()[:,:,-1,...],yb.detach().cpu().numpy()))
                )/np.size(yb.detach().cpu().numpy())
            PICP_list.append(picp)
            print(sum(PICP_list)/len(PICP_list))
            self.vallossList.append(sum(epoch_val_losses) / len(epoch_val_losses))
            mae = sum(mae_list) / len(mae_list)
            R2 = sum(R2_list) / len(R2_list)
        return self.vallossList[-1], mae, R2

    def _save(self, path):
        torch.save(self.model, path)
        return

    def load_weight(self, path):
        self.model = torch.load(path, weights_only=False)
        self.model.to(self.device)
        return

    def fit(self, train_loader, val_loader, epochs=300, savepath=None):
        self.trainlossList = []
        self.vallossList = []
        best_val_loss = math.inf
        repress = 0
        if savepath is None:
            savepath = os.path.join(self.pthpath, f'{self.cityname}_{self.time_granularity}_{epochs}epoch.pth')
            warnings.warn(f"Save path not specified, model saved to {savepath}")
        for epoch in range(epochs):
            self.now_epoch = epoch
            repress += 1
            self.model.train()
            train_loss = self._train_a_epoch(train_loader)
            self.model.eval()
            val_loss, _mae, _r2 = self._validate(val_loader)
            self.scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save(savepath)
                repress = 0
            if repress > self.early_stopping_patience:
                break
            print(f'Epoch: {epoch}; TrainLoss {train_loss:.2f}; ValLoss {val_loss:.2f}; '
                  f'lr {self.optimizer.param_groups[0]["lr"]:.6f}; '
                  f'MAE {_mae:.5f}; R2 {_r2:.5f}')

    def select_y_hat(self, y_hat_list):
        return y_hat_list

    def calculateMetricsDeterministic(self, _yTrue, _yHat, ResultDict):
        ResultDict['R2'].append(Metrics.R_squared(_yTrue, _yHat))
        ResultDict['Accuracy'].append(Metrics.get_accuracy(_yTrue, _yHat))

        _yTrue, _yHat = _yTrue.ravel(), _yHat.ravel()
        ResultDict['MAE'].append(Metrics.MAE(_yTrue, _yHat))
        ResultDict['RMSE'].append(Metrics.RMSE(_yTrue, _yHat))
        ResultDict['F1(on 0-demand)'].append(Metrics.F1_score_zeropart(_yTrue, _yHat))
        return ResultDict

    def calculateMetricsProbabilistic(self, _yTrue, _yHat, tau_list, ResultDict):
        assert len(tau_list) % 2 == 1, "Tau list should be odd, for the median 0.5 ought to be included."
        tau_percentile = [f'{int(tau*100)}' for tau in tau_list]
        _yTrue = _yTrue.ravel()
        for i in range(len(tau_list)//2):
            _mu = tau_list[-i-1] - tau_list[i]
            MPIW, PICP, CWC = Metrics.MPIW_and_PICP(_yTrue, _yHat[:,:,i,...].ravel(), _yHat[:,:,-i - 1,...].ravel(), _mu)
            ResultDict[f'MPIW@{tau_percentile[i]}-{tau_percentile[-i-1]}%'] = MPIW
            ResultDict[f'PICP@{tau_percentile[i]}-{tau_percentile[-i-1]}%'] = PICP
            ResultDict[f'CWC@{tau_percentile[i]}-{tau_percentile[-i-1]}%'] = CWC
        return ResultDict


    def test(self, test_loader, model_weight_path=None, result_path=None, verify_mode=None):
        if model_weight_path is not None:
            self.load_weight(model_weight_path)
        else:
            raise ValueError("Please specify the weight path.")

        assert verify_mode in [None, 'deterministic', 'probabilistic'], \
            "Verify mode should be either deterministic or probabilistic."

        ResultDict = {
            'MAE': [],
            'RMSE': [],
            'R2': [],
            'Accuracy': [],
            'F1(on 0-demand)': []
            }

        with torch.no_grad():
            all_y, all_y_pred = [], []
            for x_batch_list, y_batch_list in test_loader:
                _, (x_batch_list, y_batch_list, y_hat_list) = \
                    self.deliver_batch(x_batch_list, y_batch_list)
                y_batch_list = [y.cpu().detach().numpy() for y in y_batch_list]
                if isinstance(y_hat_list[0], torch.Tensor):
                    y_hat_list = [y.cpu().detach().numpy() for y in y_hat_list]
                else:
                    y_hat_list = [[y.cpu().detach().numpy() for y in y_hat] for y_hat in y_hat_list]
                all_y.append(y_batch_list)
                all_y_pred.append(y_hat_list)
            all_y, all_y_pred = np.concatenate(all_y, axis=0), np.concatenate(all_y_pred, axis=0) # concat along batch axis
            y_hat = self.select_y_hat(all_y_pred)
            _yTrue, _yHat = all_y, y_hat
            ResultDict = self.calculateMetricsDeterministic(_yTrue, _yHat, ResultDict)
            if verify_mode == 'probabilistic':
                self.calculateMetricsProbabilistic(_yTrue, all_y_pred,
                                                           [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
                                                           ResultDict)



            np.savez_compressed(result_path.replace('.json', '.npz'), y_true=all_y, y_pred=all_y_pred)

        ResultDict = {key: float(np.mean(val)) for key, val in ResultDict.items()}

        if result_path.endswith('.json'):
            # if not os.path.exists(result_path):
            #     os.makedirs(result_path)  # 没有对应文件夹则创建文件夹和文件
            # os.remove(result_path)  # 删除原有文件
            with open(result_path, 'w') as f:
                json.dump(ResultDict, f)
        else:
            raise ValueError("Result path should be a specific json-file path.")
        return ResultDict


    def inference(self, result_path=None):
        if result_path is None:
            raise ValueError("Please specify the weight path.")
        raise NotImplementedError




class SessionBqn(SessionBase):
    def __init__(self, model:nn.Module, loss_fn, optimizer=None, device=None, early_stopping_patience=None,
                 quantile_list=None, init_adj=None, env_dict=None):
        super(SessionBqn, self).__init__(model, loss_fn, optimizer, device, early_stopping_patience, env_dict)
        self.quantile_list = quantile_list
        self.loss_weight = [1.0, 1.0, 1.0]
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.optimizer_adj = optim.SGD(self.model.learnableGraph.parameters(), momentum=0.9, lr=1e-5)
        # self.optimizer_l1 = PGD(self.model.learnableGraph.parameters(), proxs=[prox_operators.prox_l1], lr=1e-5, alphas=[5e-4])
        # if self.device == torch.device("cuda"):
        #     self.optimizer_nuclear = PGD(self.model.learnableGraph.parameters(), proxs=[prox_operators.prox_nuclear_cuda], lr=1e-5, alphas=[1.5])
        # else:
        #     self.optimizer_nuclear = PGD(self.model.learnableGraph.parameters(), proxs=[prox_operators.prox_nuclear], lr=1e-5, alphas=[1.5])
        # return

    def _train_adj_a_epoch(self, train_loader):
        raise NotImplementedError
        epoch_training_losses = []
        for x_batch_list, y_batch_list in train_loader:
            # self.optimizer_adj.zero_grad()
            self.optimizer_l1.zero_grad()
            # self.optimizer_nuclear.zero_grad()
            loss_downstream, _ = self.deliver_batch(x_batch_list, y_batch_list)
            loss_l1 = torch.norm(self.model.learnableGraph(), 1)
            loss = loss_downstream + 1e0 * loss_l1
            loss.backward()
            # self.optimizer_adj.step()
            self.optimizer_l1.step()
            # self.optimizer_nuclear.step()
            epoch_training_losses.append(loss.detach().cpu().numpy().item())
        return sum(epoch_training_losses) / len(epoch_training_losses)


    def deliver_batch(self, x_batch, y_batch):
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        A_q = F.softmax(self.model.G(), dim=1)
        y_hat_list = self.model(x_batch, A_q)
        loss = 0.
        loss += self.loss_fn(y_batch, y_hat_list, self.quantile_list, 2, False)
        loss += Metrics.disMonoPenalty(y_hat_list, self.quantile_list, self.now_epoch)
        loss += 0.01 * sum([torch.norm(param)**2 for param in self.model.parameters()])
        loss += 0.1 * torch.norm(self.model.G(), p=1)
        return loss, (x_batch, y_batch, y_hat_list,)

    def _fit(self, train_loader, val_loader, epochs=300, savepath=None):
        self.trainlossList = []
        self.vallossList = []
        best_val_loss = math.inf
        repress = 0
        if savepath is None:
            savepath = os.path.join(self.pthpath, f'{self.cityname}_{self.time_granularity}_{epochs}epoch.pth')
            warnings.warn(f"Save path not specified, model saved to {savepath}")
        net_loss, adj_loss = 0., 0.
        for epoch in range(epochs):
            self.now_epoch = epoch
            repress += 1
            # 轮流训练GNN和Adj；
            if self.now_epoch % 10 < 5:
                print("GNN Training:", end=' ')
                #   训练GNN
                self.model.train()
                self.model.learnableGraph.requires_grad = False
                net_loss = self._train_a_epoch(train_loader)
            #   训练Adj
            else:
                print("Adj Training:", end=' ')
                self.model.eval()
                self.model.learnableGraph.requires_grad = True
                adj_loss = self._train_adj_a_epoch(train_loader)
            train_loss = net_loss + adj_loss
            #   验证，将所有参数都固定
            self.model.learnableGraph.requires_grad = False
            val_loss, _mae = self._validate(val_loader)
            self.scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save(savepath)
                repress = 0
            if repress > self.early_stopping_patience:
                break
            print(f'Epoch: {epoch}; TrainLoss {train_loss:.2f}; NetLoss {net_loss:.2f}; AdjLoss {adj_loss:.2f}; ValLoss {val_loss:.2f}; lr {self.optimizer.param_groups[0]["lr"]:.6f}; MAE {_mae:.5f}')


    def select_y_hat(self, y_hat_list):
        shp = y_hat_list.shape
        return y_hat_list[:,:,shp[2]//2,...]



class SessionHardParamSharingBqn(SessionBqn):
    def __init__(self, model:nn.Module, loss_fn, optimizer=None, device=None, early_stopping_patience=None,
                 quantile_list=None, init_adj=None, env_dict=None):
        super(SessionHardParamSharingBqn, self).__init__(model, loss_fn, optimizer, device, early_stopping_patience,
                                                         quantile_list, init_adj, env_dict)

    def deliver_batch(self, x_batch, y_batch):
        # timeit
        time_start = time.time()
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        y_hat_list = self.model(x_batch)
        loss = 0.
        loss += self.loss_fn(y_batch, y_hat_list, self.quantile_list, 2, False)
        # loss += Metrics.disMonoPenalty(y_hat_list, self.quantile_list, self.now_epoch)
        loss += 0.001 * sum([torch.norm(param, p=1) for param in self.model.parameters()])
        time_end = time.time()
        # print(f"Time consumed in delivering batch: {time_end - time_start}")
        return loss, (x_batch, y_batch, y_hat_list,)

























