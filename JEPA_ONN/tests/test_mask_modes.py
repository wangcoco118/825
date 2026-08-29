import unittest

import torch

from evals.intuitive_physics.train_optical import _resolve_mask_mode
from src.masks.multiblock3d import MaskCollator, UnifiedMaskCollator


def mask_config():
    return [
        {
            "aspect_ratio": [1.0, 1.0],
            "num_blocks": 1,
            "spatial_scale": [0.25, 0.25],
            "temporal_scale": [1.0, 1.0],
            "max_temporal_keep": 1.0,
            "max_keep": None,
        }
    ]


class MaskModeTests(unittest.TestCase):
    def make_collator(self, collator_type):
        return collator_type(
            cfgs_mask=mask_config(),
            crop_size=(4, 4),
            num_frames=4,
            patch_size=(2, 2),
            tubelet_size=2,
        )

    def test_mask_mode_accepts_both_values_and_defaults_to_classic(self):
        self.assertEqual(_resolve_mask_mode({}), "classic_random")
        self.assertEqual(
            _resolve_mask_mode({"mask_mode": "unified_random"}),
            "unified_random",
        )
        self.assertEqual(
            _resolve_mask_mode({"mask_mode": "classic_random"}),
            "classic_random",
        )
        with self.assertRaisesRegex(ValueError, "mask_mode"):
            _resolve_mask_mode({"mask_mode": "invalid"})

    def test_unified_random_shares_only_indices_across_batch(self):
        torch.manual_seed(0)
        collator = self.make_collator(UnifiedMaskCollator)
        masks_ctxt, masks_tgt = collator.generate_masks(4)

        self.assertEqual(len(masks_ctxt), 1)
        self.assertTrue(torch.equal(masks_ctxt[0][0], masks_ctxt[0][1]))
        self.assertTrue(torch.equal(masks_ctxt[0][1], masks_ctxt[0][2]))
        self.assertTrue(torch.equal(masks_tgt[0][0], masks_tgt[0][1]))
        self.assertTrue(torch.equal(masks_tgt[0][1], masks_tgt[0][2]))

    def test_classic_random_keeps_per_sample_mask_generation(self):
        torch.manual_seed(0)
        collator = self.make_collator(MaskCollator)
        masks_ctxt, masks_tgt = collator.generate_masks(4)

        self.assertTrue(
            any(
                not torch.equal(masks_ctxt[0][0], masks_ctxt[0][index])
                for index in range(1, 4)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(masks_tgt[0][0], masks_tgt[0][index])
                for index in range(1, 4)
            )
        )


if __name__ == "__main__":
    unittest.main()
