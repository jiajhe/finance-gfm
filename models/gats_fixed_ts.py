from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from qlib.contrib.model.pytorch_gats_ts import GATs
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path
from torch.utils.data import DataLoader, Sampler


class DailyBatchSamplerByDatetime(Sampler):
    """Yield true per-date cross sections for Qlib TSDataSampler objects."""

    def __init__(self, data_source, shuffle: bool = False):
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        index = pd.Series(np.arange(len(data_source.get_index())), index=data_source.get_index())
        self.daily_indices = [
            values.to_numpy(dtype=np.int64)
            for _, values in index.groupby(level="datetime", sort=True)
        ]

    def __iter__(self):
        order = np.arange(len(self.daily_indices))
        if self.shuffle:
            np.random.shuffle(order)
        for pos in order:
            yield self.daily_indices[int(pos)]

    def __len__(self):
        return len(self.daily_indices)


def _safe_corr_array(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return np.nan
    return float((x * y).sum() / denom)


def _safe_ir(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(arr.mean() / std)


class GRUSeqFixedSampler(Model):
    """GRU pretraining on the same TSDatasetH daily batches used by GATs."""

    def __init__(
        self,
        d_feat=157,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=60,
        lr=0.001,
        metric="loss",
        early_stop=12,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        n_jobs=0,
        **kwargs,
    ):
        self.logger = get_module_logger("GRUSeqFixedSampler")
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.loss = loss
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.n_jobs = n_jobs

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.gru_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        if self.optimizer == "adam":
            self.train_optimizer = torch.optim.Adam(self.gru_model.parameters(), lr=self.lr)
        elif self.optimizer == "gd":
            self.train_optimizer = torch.optim.SGD(self.gru_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        self.fitted = False

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def train_epoch(self, data_loader):
        self.gru_model.train()
        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            label = data[:, -1, -1].to(self.device)
            pred = self.gru_model(feature.float())
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gru_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.gru_model.eval()
        losses = []
        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            label = data[:, -1, -1].to(self.device)
            with torch.no_grad():
                pred = self.gru_model(feature.float())
                losses.append(self.loss_fn(pred, label).item())
        mean_loss = float(np.mean(losses)) if losses else 0.0
        return mean_loss, -mean_loss

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        train_loader = DataLoader(
            dl_train,
            sampler=DailyBatchSamplerByDatetime(dl_train, shuffle=True),
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dl_valid,
            sampler=DailyBatchSamplerByDatetime(dl_valid, shuffle=False),
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = copy.deepcopy(self.gru_model.state_dict())
        evals_result["train"] = []
        evals_result["valid"] = []

        self.logger.info("training...")
        self.fitted = True
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.train_epoch(train_loader)
            _, train_score = self.test_epoch(train_loader)
            _, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.gru_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.gru_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler = DailyBatchSamplerByDatetime(dl_test, shuffle=False)
        test_loader = DataLoader(dl_test, sampler=sampler, num_workers=self.n_jobs)
        index = dl_test.get_index()

        self.gru_model.eval()
        pred_series = []
        for batch_index, data in zip(sampler, test_loader):
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                pred = self.gru_model(feature.float()).detach().cpu().numpy().ravel()
            pred_series.append(pd.Series(pred, index=index[batch_index]))

        return pd.concat(pred_series).rename("score")


class GATsFixedSampler(GATs):
    """Qlib GATs with true daily batches and prediction index reconstruction."""

    def train_epoch(self, data_loader):
        self.GAT_model.train()

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            label = data[:, -1, -1].to(self.device)

            pred = self.GAT_model(feature.float())
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.GAT_model.eval()

        losses = []
        ic_values = []
        rankic_values = []

        for data in data_loader:
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(feature.float())
                loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            pred_np = pred.detach().cpu().numpy().ravel()
            label_np = label.detach().cpu().numpy().ravel()
            ic_values.append(_safe_corr_array(pred_np, label_np))
            pred_rank = pd.Series(pred_np).rank(method="average").to_numpy(dtype=np.float64)
            label_rank = pd.Series(label_np).rank(method="average").to_numpy(dtype=np.float64)
            rankic_values.append(_safe_corr_array(pred_rank, label_rank))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        metric = str(self.metric or "loss").lower()
        if metric in ("", "loss"):
            score = -mean_loss
        elif metric in ("ic", "ic_mean"):
            score = float(np.nanmean(ic_values)) if ic_values else 0.0
        elif metric == "icir":
            score = _safe_ir(ic_values)
        elif metric in ("rankic", "rank_ic", "rankic_mean", "rank_ic_mean"):
            score = float(np.nanmean(rankic_values)) if rankic_values else 0.0
        elif metric in ("rankicir", "rank_icir"):
            score = _safe_ir(rankic_values)
        else:
            raise ValueError("unknown metric `%s`" % self.metric)

        return mean_loss, score

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        train_loader = DataLoader(
            dl_train,
            sampler=DailyBatchSamplerByDatetime(dl_train, shuffle=True),
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dl_valid,
            sampler=DailyBatchSamplerByDatetime(dl_valid, shuffle=False),
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = copy.deepcopy(self.GAT_model.state_dict())
        evals_result["train"] = []
        evals_result["valid"] = []

        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path is not None:
            self.logger.info("Loading pretrained model...")
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {key: value for key, value in pretrained_model.state_dict().items() if key in model_dict}
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)
        self.logger.info("Loading pretrained model Done...")

        self.logger.info("training...")
        self.fitted = True
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            _, train_score = self.test_epoch(train_loader)
            _, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GAT_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        sampler = DailyBatchSamplerByDatetime(dl_test, shuffle=False)
        test_loader = DataLoader(dl_test, sampler=sampler, num_workers=self.n_jobs)
        index = dl_test.get_index()

        self.GAT_model.eval()
        pred_series = []
        for batch_index, data in zip(sampler, test_loader):
            data = data.squeeze()
            feature = data[:, :, 0:-1].to(self.device)
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                pred = self.GAT_model(feature.float()).detach().cpu().numpy().ravel()
            pred_series.append(pd.Series(pred, index=index[batch_index]))

        return pd.concat(pred_series).rename("score")
