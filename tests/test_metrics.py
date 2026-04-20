import unittest

import numpy as np

from eval.metrics import ic
from eval.portfolio import topk_portfolio


class MetricsTests(unittest.TestCase):
    def test_ic_metrics(self):
        preds = [np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.1, -0.2])]
        labels = [np.array([0.1, 0.21, 0.31]), np.array([0.45, 0.12, -0.18])]
        masks = [np.array([True, True, True]), np.array([True, True, True])]

        metrics = ic(preds, labels, masks)

        self.assertGreater(metrics["IC_mean"], 0.95)
        self.assertGreater(metrics["RankIC_mean"], 0.95)

    def test_topk_portfolio(self):
        preds = [np.array([0.9, 0.1, 0.3]), np.array([0.2, 0.8, 0.1])]
        labels = [np.array([0.02, -0.01, 0.00]), np.array([-0.01, 0.03, 0.00])]
        masks = [np.array([True, True, True]), np.array([True, True, True])]
        dates = ["2024-01-01", "2024-01-02"]
        instruments = [["A", "B", "C"], ["A", "B", "C"]]

        stats = topk_portfolio(preds, labels, masks, dates, k=1, instrument_lists=instruments)

        self.assertGreater(stats["annual_return"], 0.0)
        self.assertGreaterEqual(stats["turnover"], 0.0)


if __name__ == "__main__":
    unittest.main()
