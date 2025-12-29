import os, sys
sys.path.append('..')
import time

import numpy as np
import pandas as pd
import torch

import utils.Metrics as Metrics
from utils.SessionUtils import SessionHardParamSharingBqn

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from utils.Datascript import make_MultiSourceTaxiData_long_loader
from models.SoftParamSharingBqn import ST_MoE_RMQRN

for cityname in [
    #"Beijing",
    'Manhattan',
    ]:

    time_granularity = 15
    dataroot = "Datasets/"
    pthroot = "F_pth/"
    datapath = os.path.join(dataroot, cityname)
    pthpath = os.path.join(pthroot, cityname)
    env_dict = {
        'cityname': cityname,
        'time_granularity': time_granularity,
        'dataroot': dataroot,
        'pthroot': pthroot,
        'datapath': datapath,
        'pthpath': pthpath
    }

    if cityname == 'Manhattan':
        dataset_pathlist = [
            os.path.join(
                datapath,
                f'{mode}_dataset_df_{time_granularity}min.parquet')
            for mode in ['rs', 'kc']]
    else:
        dataset_pathlist = [
            os.path.join(
                datapath,
                f'{mode}_dataset_df_{time_granularity}min.parquet')
            for mode in ['rs', 'kc', 'zc']]

    X_list = []
    for _p in dataset_pathlist:
        if _p.endswith('.csv'):
            X_list.append(pd.read_csv(_p, index_col='departure_time').values)
        elif _p.endswith('.parquet'):
            X_list.append(pd.read_parquet(_p).values)
        elif _p.endswith('.npy'):
            X_list.append(np.load(_p))
        else:
            raise ValueError("Unknown data format.")
    context_info = pd.read_csv(os.path.join(datapath,f'weather_normalized.csv'), index_col='time')
    context_info.index = pd.to_datetime(context_info.index)
    
    SPACE_DIM = X_list[0].shape[1]
    task_num = len(X_list)

    trainloader, valloader, testloader = (
        make_MultiSourceTaxiData_long_loader(X_list,
                                             num_timesteps_input=24,
                                             num_timesteps_output=3,
                                             batch_size=16,
                                             split_ratio1=0.7,
                                             split_ratio2=0.8,
                                             loadmode='BQN',
                                             moe_avtivation_data=context_info)
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    prior_graph = torch.Tensor(np.load(f'Datasets/{cityname}/GraphAdj.npy')).to(device)
    # prior_graph = np.load(f'../Datasets/{cityname}/GraphAdj.npy')
    alpha_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    num_experts = 6
    Loss_Ablation = ''
    Loss_Ablation = 'LAVQ'
    # Loss_Ablation = 'Right'
    model = ST_MoE_RMQRN(space_dim=SPACE_DIM,
                         contextual_dim=context_info.values.shape[1]+8,
                         T_in=24,
                         T_out=3,
                         task_num=task_num,
                         prior_G=None,
                         quantile_list=alpha_list,
                         num_experts=num_experts
                         )
    PRETRAIN = False
    try:
        if PRETRAIN:
            model = torch.load(os.path.join(pthpath, f'{cityname}_{time_granularity}_ST_MoE_RMQRN{Loss_Ablation}_{num_experts}expert_300epoch.pth'),weights_only=False)
            import warnings
            warnings.warn('Resume Training from a pretrained weight.')
    except:
        print('Pretrained weights not found.')
        pass

    model_session = SessionHardParamSharingBqn(model, loss_fn=Metrics.JointPinballLoss, quantile_list=alpha_list,
                                               env_dict=env_dict,
                                               )
    do_train, do_test = True, True
    epochs = 1
    time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training starts at {time_start}")
    if do_train:
        # if os.path.exists(pthpath):
        #     shutil.rmtree(pthpath)
        # os.makedirs(pthpath)
        model_session.fit(trainloader, valloader, epochs=epochs,
                          savepath=os.path.join(pthpath,
                                                f'{cityname}_{time_granularity}_ST_MoE_RMQRN{Loss_Ablation}_{num_experts}expert_300epoch.pth'))

    train_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training ends at {train_time}")
    if do_test:
        model_session.test(testloader,
                           os.path.join(pthpath, f'{cityname}_{time_granularity}_ST_MoE_RMQRN{Loss_Ablation}_{num_experts}expert_300epoch.pth'),
                           f'outputs/{cityname}_ST_MoE_RMQRN{Loss_Ablation}_{time_granularity}min.json',
                           verify_mode='probabilistic')
    test_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Testing ends at {test_time}")
