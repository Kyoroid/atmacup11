import math
import unittest
import torch
from loss.rmse import RMSELoss


class TestRMSELoss(unittest.TestCase):
    def test_forward_sum(self, eps=1e-8):
        loss_module = RMSELoss(reduction="sum", eps=eps)
        input = torch.tensor(((3.0, 4.0), (4.0, 11.0)))
        target = torch.tensor(((0.0, 0.0), (-1.0, -1.0)))
        loss_actual = loss_module.forward(input, target)
        loss_expect = torch.tensor(13.92838827718412)
        self.assertAlmostEqual(loss_actual, loss_expect, delta=1e-5)

    def test_forward_mean(self, eps=1e-8):
        loss_module = RMSELoss(reduction="mean", eps=eps)
        input = torch.tensor(((3.0, 4.0), (4.0, 11.0)))
        target = torch.tensor(((0.0, 0.0), (-1.0, -1.0)))
        loss_actual = loss_module.forward(input, target)
        loss_expect = torch.tensor(6.96419413859206)
        self.assertAlmostEqual(loss_actual, loss_expect, delta=1e-5)

    def test_forward_eps(self, reduction="mean"):
        loss_module = RMSELoss(reduction=reduction, eps=10 ** 2)
        input = torch.tensor(((3.0, 4.0), (4.0, 11.0)))
        target = torch.tensor(((0.0, 0.0), (-1.0, -1.0)))
        loss_actual = loss_module.forward(input, target)
        loss_expect = torch.tensor(6.96419413859206)
        self.assertAlmostEqual(loss_actual, loss_expect, delta=10)
        self.assertNotAlmostEqual(loss_actual, loss_expect, delta=1)
