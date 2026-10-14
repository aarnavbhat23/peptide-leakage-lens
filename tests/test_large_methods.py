import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'large_study'))

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('sklearn'),'Optional study dependencies unavailable')
class LargeMethodTests(unittest.TestCase):
    def test_group_calibration_uses_maximum_per_component(self):
        from train import calibration,conformal_quantile
        scores=[1,9,2,3,4,5,6,7,8,9,10]
        groups=['A','A']+[str(i) for i in range(9)]
        q,n=calibration(scores,groups)
        self.assertEqual(n,10)
        self.assertEqual(q,10)
        self.assertGreaterEqual(q,conformal_quantile(scores))
    def test_small_calibration_set_cannot_claim_finite_90_interval(self):
        from train import conformal_quantile
        import math
        self.assertTrue(math.isinf(conformal_quantile([1,2])))
    def test_gaussian_head_has_positive_uncertainty(self):
        import torch
        from model import PotencyHead,gaussian_loss
        model=PotencyHead(5,2).eval()
        mean,sigma=model(torch.zeros(3,5),torch.tensor([0,1,2]))
        self.assertEqual(tuple(mean.shape),(3,))
        self.assertTrue((sigma>.05).all())
        self.assertTrue(torch.isfinite(gaussian_loss(mean,sigma,torch.zeros(3))))
    def test_dataset_components_cannot_cross_partitions(self):
        from build_dataset import split_indices
        groups=[i//3 for i in range(120)]
        splits=split_indices(groups)
        by_group={}
        for group,split in zip(groups,splits):
            self.assertEqual(by_group.setdefault(group,split),split)
