import unittest

import torch
import torch.nn as nn

from src.models.fsonn import FeedbackFSONN, ONNConfig
from src.models.predictor import ONNFeedbackPredictor


class RecordingONN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.slm_layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.feedback_seen = []

    def forward(self, x, feedback=None, feedback_layer_index=None):
        self.feedback_seen.append(feedback is not None)
        if feedback is not None:
            x = x + feedback
        output = self.projection(x)
        return output, output


def full_masks(batch_size=1):
    context = torch.arange(8, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    target = torch.arange(8, 1568, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    return context, target


class ONNFeedbackPredictorTests(unittest.TestCase):
    def make_predictor(self):
        return ONNFeedbackPredictor(
            embed_dim=1024,
            predictor_embed_dim=384,
            num_tokens=1568,
            num_chunks=8,
            chunk_tokens=196,
            output_mlp_hidden_dim=384,
            onn_core=RecordingONN(384),
        )

    def test_forward_has_fixed_canvas_and_1024_output(self):
        predictor = self.make_predictor()
        context = torch.randn(1, 8, 1024)
        target = torch.randn(1, 1560, 1024)
        masks_ctxt, masks_tgt = full_masks()

        output = predictor(context, target, masks_ctxt, masks_tgt)

        self.assertEqual(tuple(output.shape), (1, 1560, 1024))
        self.assertEqual(predictor.last_trace["onn_input_shape"], (1, 1568, 384))
        self.assertEqual(predictor.last_trace["chunk_shape"], (1, 8, 196, 384))
        self.assertEqual(predictor.last_trace["dense_output_shape"], (1, 1568, 384))
        self.assertEqual(predictor.last_trace["pred_tgt_384_shape"], (1, 1560, 384))
        self.assertEqual(predictor.last_trace["pred_tgt_1024_shape"], (1, 1560, 1024))

    def test_mask_token_mode_does_not_read_real_target(self):
        predictor = self.make_predictor()
        context = torch.randn(1, 8, 1024)
        target_a = torch.randn(1, 1560, 1024)
        target_b = torch.randn(1, 1560, 1024)
        masks_ctxt, masks_tgt = full_masks()

        output_a = predictor(context, target_a, masks_ctxt, masks_tgt)
        output_b = predictor(context, target_b, masks_ctxt, masks_tgt)

        self.assertTrue(torch.equal(output_a, output_b))

    def test_feedback_starts_empty_then_uses_previous_chunk(self):
        predictor = self.make_predictor()
        context = torch.randn(1, 8, 1024)
        masks_ctxt, masks_tgt = full_masks()

        predictor(context, None, masks_ctxt, masks_tgt)

        self.assertEqual(
            predictor.onn_core.feedback_seen,
            [False, True, True, True, True, True, True, True],
        )

    def test_invalid_fixed_token_index_is_rejected(self):
        predictor = self.make_predictor()
        context = torch.randn(1, 8, 1024)
        masks_ctxt, masks_tgt = full_masks()
        masks_tgt[:, -1] = 1568

        with self.assertRaisesRegex(ValueError, "1567"):
            predictor(context, None, masks_ctxt, masks_tgt)

    def test_position_embedding_is_frozen_buffer_and_default_feedback_is_layer_two(self):
        predictor = self.make_predictor()

        self.assertNotIn("predictor_pos_embed", dict(predictor.named_parameters()))
        self.assertIn("predictor_pos_embed", dict(predictor.named_buffers()))
        self.assertEqual(predictor.feedback_layer_index, 2)

    def test_feedback_onn_has_single_intensity_readout(self):
        config = ONNConfig.from_mapping(
            {
                "input_dim": 2,
                "output_dim": 2,
                "num_slm_layers": 4,
                "chunk_tokens": 2,
                "grid_height": 2,
                "grid_width": 2,
                "feedback_mode": "fixed_middle",
                "feedback_layer_index": 2,
                "pixel_pitch_um": 8.0,
                "wavelength_nm": 532.0,
                "slm_intervals_um": [8.0, 8.0, 8.0],
                "input_to_first_slm_um": 8.0,
                "last_slm_to_detector_um": 8.0,
                "asm_padding_factor": 1.0,
                "learnable_intensity_offset": True,
                "use_differential_detector": False,
            }
        )
        model = FeedbackFSONN(config)
        names = [name for name, _ in model.named_parameters()]
        forbidden = (
            "positive_field",
            "negative_field",
            "positive_gain_raw",
            "negative_gain_raw",
            "differential_detector_gap_um",
            "detector_split_ratio",
        )
        self.assertFalse(any(any(token in name for token in forbidden) for name in names))
        output = model(torch.randn(1, 2, 2))
        self.assertEqual(tuple(output.shape), (1, 2, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_onn_and_output_parameters_receive_finite_gradients(self):
        predictor = self.make_predictor()
        context = torch.randn(1, 8, 1024)
        masks_ctxt, masks_tgt = full_masks()

        loss = predictor(context, None, masks_ctxt, masks_tgt).square().mean()
        loss.backward()

        trainable = [
            parameter for parameter in predictor.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.grad is not None for parameter in trainable))
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in trainable)
        )


if __name__ == "__main__":
    unittest.main()
