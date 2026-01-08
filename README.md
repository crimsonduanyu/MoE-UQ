# ST-MoE-RMQRN (Code Repo)

REFACTORED CODES ARE AVAILABLE AT: https://github.com/crimsonduanyu/MoE-UQ-Refactored

This repository contains the reference implementation for the paper:

**“Uncertainty quantification for joint demand prediction of multi-modal ride-sourcing services using spatiotemporal Mixture-of-Expert neural network”**

It trains/tests a **spatiotemporal Mixture-of-Experts** model for **joint (multi-task) demand forecasting** with **probabilistic (quantile) uncertainty quantification**.

---

## 1) Project Structure

- **Training / Testing entry**
  - [SessionST-MoE-RMQRN.py](SessionST-MoE-RMQRN.py): main script to build dataloaders, create the model, train, and test.

- **Models**
  - [models/SoftParamSharingBqn.py](models/SoftParamSharingBqn.py)
    - [`models.SoftParamSharingBqn.ST_MoE_RMQRN`](models/SoftParamSharingBqn.py): spatiotemporal MoE model (multi-task, quantile outputs).

- **Data**
  - [Datasets/](Datasets/)
    - City data folders: `Datasets/Manhattan/`, `Datasets/Beijing/`
      - Expected files like `rs_dataset_df_15min.parquet`, `kc_dataset_df_15min.parquet`, `zc_dataset_df_15min.parquet` (Beijing only).
    - Weather-related preprocessing resources in [Datasets/WeatherData/](Datasets/WeatherData/)
    - Analysis notebook: [Datasets/DataDistribution.ipynb](Datasets/DataDistribution.ipynb) (exports `number_distribution_*.csv`)

- **Dataloaders / preprocessing**
  - [utils/Datascript.py](utils/Datascript.py)
    - [`utils.Datascript.make_MultiSourceTaxiData_long_loader`](utils/Datascript.py): builds train/val/test loaders for multi-source datasets.
    - [`utils.Datascript.MultiSourceTaxiData_long`](utils/Datascript.py): long-history dataset mode (used by `loadmode='BQN'`).

- **Training session + evaluation**
  - [utils/SessionUtils.py](utils/SessionUtils.py)
    - [`utils.SessionUtils.SessionHardParamSharingBqn`](utils/SessionUtils.py): training/testing loop wrapper used in the entry script.
    - [`utils.SessionUtils.SessionBase.calculateMetricsProbabilistic`](utils/SessionUtils.py): probabilistic evaluation (PICP/MPIW/CWC etc.).
  - [utils/Metrics.py](utils/Metrics.py): loss and metrics utilities (used by the session).

- **Outputs**
  - [outputs/](outputs/): JSON metrics dumped by test phase.
  - [F_pth/](F_pth/): saved `.pth` weights by city.

---

## 2) Environment

This repo is **Python + PyTorch**.

Install the common dependencies (example):

````bash
pip install numpy pandas scipy torch matplotlib timm

````

if you find our work useful, please cite:
````
@article{LIU2026105507,
title = {Uncertainty quantification for joint demand prediction of multi-modal ride-sourcing services using spatiotemporal Mixture-of-Expert neural network},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {184},
pages = {105507},
year = {2026},
issn = {0968-090X},
doi = {https://doi.org/10.1016/j.trc.2025.105507},
url = {https://www.sciencedirect.com/science/article/pii/S0968090X2500511X},
author = {Xiaobing Liu and Yu Duan and Yangli-ao Geng and Yun Wang and Qingyong LI and Xuedong Yan and Ziyou Gao}
}
