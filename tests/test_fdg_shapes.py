import unittest

import torch

from models.fdg import FDG


class FDGShapeTests(unittest.TestCase):
    def test_shapes_and_masking(self):
        torch.manual_seed(0)
        model = FDG(d_in=5, rank=3, tau=1.0)
        X = torch.randn(2, 4, 5)
        mask = torch.tensor([[True, True, False, True], [True, False, False, False]])

        A, S, R = model(X, mask=mask)

        self.assertEqual(tuple(A.shape), (2, 4, 4))
        self.assertEqual(tuple(S.shape), (2, 4, 3))
        self.assertEqual(tuple(R.shape), (2, 4, 3))
        self.assertTrue(torch.allclose(A[0, 2], torch.zeros(4), atol=1e-6))
        self.assertTrue(torch.allclose(A[0, :, 2], torch.zeros(4), atol=1e-6))

        row_sums = A[0, mask[0]].sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
