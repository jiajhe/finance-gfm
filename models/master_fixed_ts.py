from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from qlib.contrib.model.pytorch_master_ts import MASTERModel
from qlib.data.dataset.handler import DataHandlerLP


class DailyBatchSamplerByDatetime(Sampler):
    """Yield true per-date cross sections even when TSDataSampler is not date-sorted."""

    def __init__(self, data_source, shuffle: bool = False):
        self.data_source = data_source
        self.shuffle = shuffle
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


class MASTERModelFixedSampler(MASTERModel):
    """MASTER with a sampler/index fix for Qlib's instrument-major TSDataSampler."""

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerByDatetime(data, shuffle)
        return DataLoader(data, sampler=sampler, drop_last=drop_last)

    def predict(self, dataset, use_pretrained=False):
        if use_pretrained:
            self.load_param(f"{self.save_path}{self.save_prefix}master_{self.seed}.pkl")
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        sampler = DailyBatchSamplerByDatetime(dl_test, shuffle=False)
        test_loader = DataLoader(dl_test, sampler=sampler, drop_last=False)
        index = dl_test.get_index()

        pred_series = []
        self.model.eval()
        for batch_index, data in zip(sampler, test_loader):
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy().ravel()
            pred_series.append(pd.Series(pred, index=index[batch_index]))

        return pd.concat(pred_series).to_frame()
