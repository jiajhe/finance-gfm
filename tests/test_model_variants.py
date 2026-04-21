import unittest

import pandas as pd
import torch

from data.qlib_loader import _apply_feature_transforms, pad_collate
from models import build_model
from models import FDGPriorRegressor
from models.blocks import FeatureBottleneck
from models.baselines.mlp import GraphResidualMLP, MLPBaseline, TemporalGraphResidualMLP
from models.fdg_regularized import RegularizedFDGRegressor
from models.fdg_slowfast import SlowFastTemporalFDGRegressor
from models.fdg_sparse import SparseRollingFDGRegressor
from models.fdg_temporal import TemporalFDGRegressor
from models.temporal_graph import EntropyGraphRegressor, EntropyStockGraph, FDGAuxGraphRegressor, RollingCorrelationGraph
from train.loss import build_loss, trim_extreme_mask


class ModelVariantTests(unittest.TestCase):
    def test_feature_rank_transform_selected_columns(self):
        features = pd.DataFrame(
            {
                "IMXD60": [10.0, 20.0, 30.0],
                "KEEP": [1.0, 3.0, 2.0],
                "CORR60": [5.0, 1.0, 3.0],
            },
            index=["A", "B", "C"],
        )

        transformed = _apply_feature_transforms(
            features,
            rank_features=["IMXD60", "CORR60"],
        )

        self.assertEqual(list(transformed.columns), ["IMXD60", "KEEP", "CORR60"])
        self.assertTrue(
            torch.allclose(
                torch.tensor(transformed["IMXD60"].to_numpy(), dtype=torch.float32),
                torch.tensor([-1 / 3, 1 / 3, 1.0], dtype=torch.float32),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(transformed["CORR60"].to_numpy(), dtype=torch.float32),
                torch.tensor([1.0, -1 / 3, 1 / 3], dtype=torch.float32),
                atol=1e-6,
            )
        )
        self.assertEqual(list(transformed["KEEP"]), [1.0, 3.0, 2.0])

    def test_feature_drop_transform_removes_columns(self):
        features = pd.DataFrame(
            {
                "IMXD60": [1.0, 2.0],
                "CNTN60": [3.0, 4.0],
                "KEEP": [5.0, 6.0],
            }
        )

        transformed = _apply_feature_transforms(
            features,
            drop_features=["IMXD60", "CNTN60"],
        )

        self.assertEqual(list(transformed.columns), ["KEEP"])

    def test_fused_fdg_gate_shapes(self):
        torch.manual_seed(0)
        model = FDGPriorRegressor(
            d_in=5,
            rank=3,
            d_hidden=7,
            prior_centroids=torch.randn(3, 5),
            prior_mean=torch.zeros(5),
            prior_scale=torch.ones(5),
            prior_fusion="gate",
            bottleneck_dim=3,
        )
        X = torch.randn(2, 4, 5)
        mask = torch.tensor([[True, True, False, True], [True, True, True, False]])

        y_hat, graph = model(X, mask=mask, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (2, 4))
        self.assertEqual(tuple(graph["A"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["A_prior"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["C"].shape), (2, 4, 3))
        row_sums = graph["A"][0, mask[0]].sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_fused_fdg_shared_shapes(self):
        torch.manual_seed(1)
        model = FDGPriorRegressor(
            d_in=6,
            rank=4,
            d_hidden=8,
            prior_centroids=torch.randn(4, 6),
            prior_mean=torch.zeros(6),
            prior_scale=torch.ones(6),
            prior_fusion="shared",
            bottleneck_dim=4,
        )
        X = torch.randn(1, 5, 6)
        mask = torch.tensor([[True, True, True, False, True]])

        y_hat, graph = model(X, mask=mask, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (1, 5))
        self.assertEqual(tuple(graph["A_fdg"].shape), (1, 5, 5))
        self.assertEqual(tuple(graph["A_prior"].shape), (1, 5, 5))
        self.assertGreaterEqual(float(graph["mix"]), 0.0)
        self.assertLessEqual(float(graph["mix"]), 1.0)

    def test_mlp_residual_bottleneck_shape(self):
        torch.manual_seed(2)
        model = MLPBaseline(d_in=5, d_hidden=9, dropout=0.1, bottleneck_dim=3, residual_layers=2)
        X = torch.randn(3, 4, 5)
        mask = torch.tensor(
            [
                [True, True, False, True],
                [True, True, True, True],
                [True, False, False, False],
            ]
        )

        y_hat = model(X, mask)

        self.assertEqual(tuple(y_hat.shape), (3, 4))
        self.assertTrue(torch.allclose(y_hat[0, 2], torch.tensor(0.0), atol=1e-6))

    def test_stacked_feature_bottleneck_shape(self):
        torch.manual_seed(2)
        module = FeatureBottleneck(d_in=5, bottleneck_dim=[4, 3, 4], dropout=0.1)
        X = torch.randn(2, 4, 5)
        mask = torch.tensor([[True, True, False, True], [True, True, True, False]])

        out = module(X, mask=mask)

        self.assertEqual(tuple(out.shape), (2, 4, 5))
        self.assertTrue(torch.allclose(out[0, 2], torch.zeros(5), atol=1e-6))

    def test_fdg_roll_variant_shapes(self):
        torch.manual_seed(3)
        model = FDGAuxGraphRegressor(
            d_in=5,
            rank=3,
            d_hidden=7,
            aux_graph=RollingCorrelationGraph(topk=2),
            bottleneck_dim=3,
            final_topk=2,
        )
        X = torch.randn(2, 4, 5)
        history = torch.randn(2, 4, 20)
        mask = torch.tensor([[True, True, True, False], [True, True, False, True]])

        y_hat, graph = model(X, mask=mask, history=history, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (2, 4))
        self.assertEqual(tuple(graph["A"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["A_aux"].shape), (2, 4, 4))
        row_sums = graph["A"][0, mask[0]].sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_entropy_graph_variant_shapes(self):
        torch.manual_seed(4)
        model = EntropyGraphRegressor(
            d_in=6,
            d_hidden=8,
            bottleneck_dim=4,
            graph=EntropyStockGraph(topk=2, num_bins=6),
        )
        X = torch.randn(1, 5, 6)
        history = torch.randn(1, 5, 20)
        mask = torch.tensor([[True, True, True, False, True]])

        y_hat, graph = model(X, mask=mask, history=history, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (1, 5))
        self.assertEqual(tuple(graph["A"].shape), (1, 5, 5))
        row_sums = graph["A"][0, mask[0]].sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_fdg_temporal_variant_shapes(self):
        torch.manual_seed(7)
        model = TemporalFDGRegressor(
            d_in=6,
            rank=3,
            d_hidden=8,
            dropout=0.1,
            bottleneck_dim=4,
            conv_channels=8,
            temporal_kernel_size=3,
        )
        X = torch.randn(2, 5, 6)
        history = torch.randn(2, 5, 20)
        mask = torch.tensor([[True, True, True, False, True], [True, True, False, True, True]])

        y_hat, graph = model(X, mask=mask, history=history, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (2, 5))
        self.assertEqual(tuple(graph["A"].shape), (2, 5, 5))
        self.assertEqual(tuple(graph["X_history"].shape), (2, 5, 6))

    def test_fdg_sparse_variant_shapes(self):
        torch.manual_seed(8)
        model = SparseRollingFDGRegressor(
            d_in=5,
            rank=3,
            d_hidden=7,
            bottleneck_dim=3,
            fdg_topk=2,
            roll_topk=2,
            final_topk=2,
            edge_dropout=0.1,
        )
        X = torch.randn(2, 4, 5)
        history = torch.randn(2, 4, 20)
        mask = torch.tensor([[True, True, True, False], [True, True, False, True]])

        y_hat, graph = model(X, mask=mask, history=history, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (2, 4))
        self.assertEqual(tuple(graph["A"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["A_roll"].shape), (2, 4, 4))

    def test_fdg_slowfast_variant_shapes(self):
        torch.manual_seed(10)
        model = SlowFastTemporalFDGRegressor(
            d_in=5,
            rank=3,
            d_hidden=7,
            bottleneck_dim=[4, 3, 4],
            conv_channels=8,
            temporal_kernel_size=3,
            graph_smooth_weight=1e-3,
            assignment_smooth_weight=1e-3,
        )
        X = torch.randn(2, 4, 5)
        history = torch.randn(2, 4, 20)
        mask = torch.tensor([[True, True, True, False], [True, True, False, True]])

        y_hat, graph = model(X, mask=mask, history=history, return_graph=True)
        reg_loss = model.regularization_loss()

        self.assertEqual(tuple(y_hat.shape), (2, 4))
        self.assertEqual(tuple(graph["A"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["A_slow"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["X_fast"].shape), (2, 4, 5))
        self.assertTrue(torch.isfinite(reg_loss))

    def test_fdg_regularized_variant_has_reg_loss(self):
        torch.manual_seed(9)
        model = RegularizedFDGRegressor(
            d_in=5,
            rank=3,
            d_hidden=7,
            bottleneck_dim=3,
            adjacency_topk=2,
            core_reg_weight=1e-4,
            graph_entropy_weight=1e-3,
            assignment_entropy_weight=1e-3,
        )
        X = torch.randn(2, 4, 5)
        mask = torch.tensor([[True, True, True, False], [True, True, False, True]])

        y_hat, graph = model(X, mask=mask, return_graph=True)
        reg_loss = model.regularization_loss()

        self.assertEqual(tuple(y_hat.shape), (2, 4))
        self.assertEqual(tuple(graph["A"].shape), (2, 4, 4))
        self.assertTrue(torch.isfinite(reg_loss))
        self.assertGreaterEqual(float(reg_loss), 0.0)

    def test_mlp_graph_plugin_shapes(self):
        torch.manual_seed(5)
        model = GraphResidualMLP(
            d_in=5,
            d_hidden=8,
            dropout=0.1,
            rank=3,
            bottleneck_dim=3,
            residual_layers=2,
            graph_layers=1,
            graph_kind="fdg_roll",
            roll_topk=2,
            final_topk=2,
        )
        X = torch.randn(2, 4, 5)
        history = torch.randn(2, 4, 20)
        mask = torch.tensor([[True, True, True, False], [True, True, False, True]])

        y_hat, graph = model(X, mask=mask, history=history, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (2, 4))
        self.assertEqual(tuple(graph["A"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["A_fdg"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["A_aux"].shape), (2, 4, 4))

    def test_temporal_graph_plugin_shapes(self):
        torch.manual_seed(6)
        model = TemporalGraphResidualMLP(
            d_in=5,
            d_hidden=8,
            dropout=0.1,
            rank=3,
            bottleneck_dim=3,
            residual_layers=1,
            graph_layers=1,
            graph_kind="fdg_roll",
            roll_topk=2,
            final_topk=2,
            history_window=20,
            temporal_layers=1,
            temporal_heads=4,
        )
        X = torch.randn(2, 4, 5)
        history = torch.randn(2, 4, 20)
        mask = torch.tensor([[True, True, True, False], [True, True, False, True]])

        y_hat, graph = model(X, mask=mask, history=history, return_graph=True)

        self.assertEqual(tuple(y_hat.shape), (2, 4))
        self.assertEqual(tuple(graph["A"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["A_fdg"].shape), (2, 4, 4))
        self.assertEqual(tuple(graph["gate"].shape), (2, 1, 8))

    def test_drop_extreme_loss_keeps_signal(self):
        y = torch.tensor([[0.0, 1.0, 2.0, 3.0, 50.0, -40.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        mask = torch.ones_like(y, dtype=torch.bool)
        trimmed = trim_extreme_mask(y=y, mask=mask, drop_pct=0.2)
        self.assertEqual(int(trimmed.sum().item()), 10)

        preds = y.clone()
        loss_fn = build_loss(loss_name="ic_wpcc", drop_extreme_pct=0.2, wpcc_weight=0.2)
        loss = loss_fn(preds, y, mask)
        self.assertTrue(torch.isfinite(loss))

    def test_mse_loss_and_pad_collate_raw_labels(self):
        loss_fn = build_loss(loss_name="mse")
        preds = torch.tensor([[0.0, 1.0, 2.0]])
        y = torch.tensor([[0.5, 1.5, 2.5]])
        mask = torch.tensor([[True, True, False]])
        loss = loss_fn(preds, y, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

        mixed_loss_fn = build_loss(loss_name="mse_ic_wpcc", ic_weight=0.1, wpcc_weight=0.05)
        mixed_loss = mixed_loss_fn(preds, y, mask)
        self.assertTrue(torch.isfinite(mixed_loss))

        batch = [
            (
                torch.randn(2, 5),
                torch.tensor([1.0, 2.0]),
                torch.tensor([0.1, 0.2]),
                torch.tensor([True, True]),
                "2020-01-01",
                ["A", "B"],
                None,
            ),
            (
                torch.randn(3, 5),
                torch.tensor([3.0, 4.0, 5.0]),
                torch.tensor([0.3, 0.4, 0.5]),
                torch.tensor([True, True, True]),
                "2020-01-02",
                ["C", "D", "E"],
                None,
            ),
        ]
        X, y_batch, raw_y_batch, mask_batch, dates, instruments, history = pad_collate(batch)
        self.assertEqual(tuple(X.shape), (2, 3, 5))
        self.assertEqual(tuple(y_batch.shape), (2, 3))
        self.assertEqual(tuple(raw_y_batch.shape), (2, 3))
        self.assertEqual(tuple(mask_batch.shape), (2, 3))
        self.assertIsNone(history)
        self.assertEqual(dates, ["2020-01-01", "2020-01-02"])
        self.assertEqual(instruments[0], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
