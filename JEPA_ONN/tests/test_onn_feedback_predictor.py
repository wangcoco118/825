import inspect
import unittest

import torch
import torch.nn as nn

from src.models.fsonn import FeedbackFSONN, ONNConfig, PhaseSLM
from src.models.predictor import ONNFeedbackPredictor


class RecordingONN(nn.Module):
    def __init__(self, dim, num_slm_layers=4):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.slm_layers = nn.ModuleList(
            [nn.Identity() for _ in range(num_slm_layers)]
        )
        self.feedback_values = []
        self.feedback_layer_indices_seen = []
        self.outputs = []

    def forward(
        self,
        x,
        feedback=None,
        feedback_layer_index=None,
        feedback_layer_indices=None,
    ):
        self.feedback_values.append(feedback)
        indices = feedback_layer_indices
        if indices is None and feedback_layer_index is not None:
            indices = (feedback_layer_index,)
        self.feedback_layer_indices_seen.append(
            None if indices is None else tuple(indices)
        )
        output = self.projection(x)
        self.outputs.append(output)
        return output, output


def full_masks(batch_size=1):
    context = torch.arange(8, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    target = torch.arange(8, 1568, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    return context, target


def small_onn_values():
    return {
        "input_dim": 2,
        "output_dim": 2,
        "num_slm_layers": 4,
        "chunk_tokens": 2,
        "grid_height": 2,
        "grid_width": 2,
        "feedback_mode": "fixed_middle",
        "feedback_layer_index": 2,
        "feedback_phase_max_rad": 0.75,
        "feedback_gain_init": 1.0,
        "pixel_pitch_um": 8.0,
        "wavelength_nm": 532.0,
        "slm_intervals_um": [8.0, 8.0, 8.0],
        "input_to_first_slm_um": 8.0,
        "last_slm_to_detector_um": 8.0,
        "asm_padding_factor": 1.0,
        "learnable_intensity_offset": True,
        "use_differential_detector": False,
    }


def small_onn_config(**overrides):
    values = small_onn_values()
    values.update(overrides)
    return ONNConfig.from_mapping(values)


def small_multi_onn_config(gain_mode, gain_init=1.0, layer_indices=(2, 3, 4)):
    values = small_onn_values()
    values.update(
        {
            "num_slm_layers": 5,
            "slm_intervals_um": [8.0, 8.0, 8.0, 8.0],
            "feedback_layer_mode": "multi",
            "feedback_layer_indices": list(layer_indices),
            "feedback_gain_mode": gain_mode,
            "feedback_gain_init": gain_init,
            "feedback_gain_epsilon": 1.0e-6,
        }
    )
    values.pop("feedback_layer_index")
    return ONNConfig.from_mapping(values)


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

    def test_partial_masks_are_allowed_and_report_missing_count(self):
        predictor = self.make_predictor()
        context = torch.randn(1, 8, 1024)
        masks_ctxt = torch.arange(8, dtype=torch.long).unsqueeze(0)
        masks_tgt = torch.arange(8, 1008, dtype=torch.long).unsqueeze(0)

        output = predictor(context, None, masks_ctxt, masks_tgt)

        self.assertEqual(tuple(output.shape), (1, 1000, 1024))
        self.assertEqual(predictor.last_trace["covered_count"], 1008)
        self.assertEqual(predictor.last_trace["missing_count"], 560)

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

        feedback_values = predictor.onn_core.feedback_values
        self.assertIsNone(feedback_values[0])
        self.assertEqual(
            [value is not None for value in feedback_values],
            [False, True, True, True, True, True, True, True],
        )
        for chunk_index in range(1, 8):
            self.assertIs(
                feedback_values[chunk_index],
                predictor.onn_core.outputs[chunk_index - 1],
            )

    def test_overlapping_masks_are_rejected(self):
        predictor = self.make_predictor()
        context = torch.randn(1, 2, 1024)
        masks_ctxt = torch.tensor([[0, 1]], dtype=torch.long)
        masks_tgt = torch.tensor([[1, 2]], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "overlap or duplicate"):
            predictor(context, None, masks_ctxt, masks_tgt)

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

    def test_dynamic_phase_configuration_and_parameter_contract(self):
        config = small_onn_config()

        self.assertEqual(config.feedback_mode, "fixed_middle_phase")
        self.assertEqual(config.feedback_phase_max_rad, 0.75)
        self.assertEqual(config.feedback_gain_init, 1.0)
        model = FeedbackFSONN(config)
        self.assertEqual(model.feedback_gain_raw.numel(), 1)
        gain = model._positive_parameter(model.feedback_gain_raw)
        self.assertAlmostEqual(gain.item(), 1.0, places=6)
        self.assertEqual(list(model.feedback_norm.parameters()), [])
        feedback_names = [
            name for name, _ in model.named_parameters()
            if "feedback_gain" in name
        ]
        self.assertEqual(feedback_names, ["feedback_gain_raw"])
        forbidden = ("feedback_bias", "phase_bias", "feedback_shift")
        self.assertFalse(
            any(any(token in name for token in forbidden) for name, _ in model.named_parameters())
        )

    def test_dynamic_feedback_phase_is_bounded_zero_centered_and_differentiable(self):
        config = small_onn_config()
        model = FeedbackFSONN(config)
        feedback = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])

        phase = model._feedback_to_phase(feedback)

        self.assertEqual(tuple(phase.shape), (1, 2, 2))
        self.assertTrue(torch.isfinite(phase).all())
        self.assertLessEqual(phase.abs().max().item(), 0.75 + 1e-6)
        self.assertTrue(torch.equal(model._feedback_to_phase(torch.zeros_like(feedback)), torch.zeros_like(feedback)))
        phase.square().mean().backward()
        self.assertIsNotNone(model.feedback_gain_raw.grad)
        self.assertTrue(torch.isfinite(model.feedback_gain_raw.grad).all())
        self.assertNotEqual(model.feedback_gain_raw.grad.item(), 0.0)

    def test_phase_slm_adds_feedback_as_phase_modulation(self):
        self.assertIn("phase_delta", inspect.signature(PhaseSLM.forward).parameters)
        slm = PhaseSLM(2, 2)
        real = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        imag = torch.tensor([[[0.5, -0.5], [1.0, -1.0]]])
        field = torch.complex(real, imag)
        phase_delta = torch.tensor([[[0.1, -0.2], [0.3, -0.4]]])
        base_phase = 2.0 * torch.pi * torch.sigmoid(slm.phase_logits)
        expected = field * torch.exp(1j * (base_phase + phase_delta))

        actual = slm(field, phase_delta=phase_delta)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_none_feedback_and_zero_feedback_are_equivalent(self):
        model = FeedbackFSONN(small_onn_config())
        slot = torch.tensor([[[0.2, -0.3], [0.4, -0.1]]])

        without_feedback, debug_without = model._propagate_slot(slot, feedback=None)
        with_zero_feedback, debug_zero = model._propagate_slot(
            slot,
            feedback=torch.zeros_like(slot),
        )

        self.assertTrue(torch.allclose(without_feedback, with_zero_feedback, atol=1e-6, rtol=1e-6))
        self.assertFalse(debug_without["feedback_used"])
        self.assertTrue(debug_zero["feedback_used"])
        self.assertTrue(torch.equal(debug_zero["feedback_phase"], torch.zeros_like(slot)))

    def test_invalid_dynamic_feedback_configuration_is_rejected(self):
        for key, value in (
            ("feedback_phase_max_rad", 0.0),
            ("feedback_phase_max_rad", -1.0),
            ("feedback_gain_init", 0.0),
            ("feedback_gain_init", -1.0),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    small_onn_config(**{key: value})



    def test_predictor_multi_mode_passes_layer_indices_without_changing_chunk_order(self):
        self.assertIn(
            "feedback_layer_indices",
            inspect.signature(ONNFeedbackPredictor.__init__).parameters,
        )
        core = RecordingONN(384, num_slm_layers=5)
        predictor = ONNFeedbackPredictor(
            embed_dim=1024,
            predictor_embed_dim=384,
            num_tokens=1568,
            num_chunks=8,
            chunk_tokens=196,
            output_mlp_hidden_dim=384,
            feedback_layer_mode="multi",
            feedback_layer_indices=[2, 3, 4],
            feedback_gain_mode="shared",
            onn_core=core,
        )
        context = torch.randn(1, 1, 1024)
        masks_ctxt = torch.tensor([[0]], dtype=torch.long)
        masks_tgt = torch.tensor([[1567]], dtype=torch.long)

        output = predictor(context, None, masks_ctxt, masks_tgt)

        self.assertEqual(tuple(output.shape), (1, 1, 1024))
        self.assertIsNone(core.feedback_values[0])
        self.assertEqual(
            [value is not None for value in core.feedback_values],
            [False, True, True, True, True, True, True, True],
        )
        self.assertEqual(
            core.feedback_layer_indices_seen,
            [(2, 3, 4)] * 8,
        )
        self.assertEqual(predictor.feedback_layer_mode, "multi")
        self.assertEqual(predictor.feedback_layer_indices, (2, 3, 4))
        self.assertIsNone(predictor.feedback_layer_index)
        self.assertEqual(predictor.feedback_gain_mode, "shared")
        self.assertEqual(predictor.last_trace["feedback_layer_mode"], "multi")
        self.assertEqual(
            predictor.last_trace["feedback_layer_indices"],
            (2, 3, 4),
        )


    def test_legacy_single_config_preserves_scalar_gain_and_formula(self):
        config = small_onn_config()
        self.assertTrue(hasattr(config, "feedback_layer_mode"))
        self.assertEqual(config.feedback_layer_mode, "single")
        self.assertEqual(config.feedback_layer_index, 2)
        self.assertIsNone(config.feedback_layer_indices)
        self.assertIsNone(config.feedback_gain_mode)
        model = FeedbackFSONN(config)
        self.assertEqual(tuple(model.feedback_gain_raw.shape), ())
        feedback = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
        normalized = model.feedback_norm(feedback)
        gain = model._effective_feedback_gains()
        expected = config.feedback_phase_max_rad * torch.tanh(gain * normalized)
        actual = model._feedback_to_phases(feedback)
        self.assertEqual(tuple(actual), (2,))
        self.assertTrue(torch.equal(actual[2], expected))

    def test_multi_shared_uses_one_scalar_and_equal_layer_phases(self):
        config = small_multi_onn_config("shared")
        self.assertEqual(config.feedback_layer_mode, "multi")
        self.assertEqual(config.feedback_layer_indices, (2, 3, 4))
        self.assertEqual(config.feedback_gain_mode, "shared")
        model = FeedbackFSONN(config)
        self.assertEqual(tuple(model.feedback_gain_raw.shape), ())
        feedback = torch.randn(1, 2, 2)

        phases = model._feedback_to_phases(feedback)

        self.assertEqual(tuple(phases), (2, 3, 4))
        self.assertTrue(torch.equal(phases[2], phases[3]))
        self.assertTrue(torch.equal(phases[3], phases[4]))
        self.assertLessEqual(
            max(value.abs().max().item() for value in phases.values()),
            config.feedback_phase_max_rad + 1e-6,
        )

    def test_multi_independent_gains_are_not_normalized_and_are_layer_local(self):
        config = small_multi_onn_config(
            "independent",
            gain_init=[0.5, 1.5, 3.0],
        )
        model = FeedbackFSONN(config)
        self.assertEqual(tuple(model.feedback_gain_raw.shape), (3,))
        gains = model._effective_feedback_gains()
        self.assertTrue(
            torch.allclose(gains, torch.tensor([0.5, 1.5, 3.0]), atol=1e-6)
        )
        self.assertAlmostEqual(gains.sum().item(), 5.0, places=6)
        feedback = torch.randn(1, 2, 2)
        before = model._feedback_to_phases(feedback)

        with torch.no_grad():
            model.feedback_gain_raw[1].add_(0.75)
        after = model._feedback_to_phases(feedback)

        self.assertTrue(torch.equal(before[2], after[2]))
        self.assertFalse(torch.equal(before[3], after[3]))
        self.assertTrue(torch.equal(before[4], after[4]))
        self.assertTrue(
            all(
                phase.abs().max() <= config.feedback_phase_max_rad + 1e-6
                for phase in after.values()
            )
        )

    def test_layer_mode_configuration_is_strictly_mutually_exclusive(self):
        invalid = []
        values = small_onn_values()
        values.update(
            feedback_layer_mode="single",
            feedback_layer_indices=[2, 3],
        )
        invalid.append(values)
        values = small_onn_values()
        values.update(
            feedback_layer_mode="single",
            feedback_gain_mode="shared",
        )
        invalid.append(values)
        values = small_onn_values()
        values.pop("feedback_layer_index")
        values.update(
            feedback_layer_mode="multi",
            feedback_layer_indices=[2],
            feedback_gain_mode="shared",
        )
        invalid.append(values)
        values = small_onn_values()
        values.pop("feedback_layer_index")
        values.update(
            feedback_layer_mode="multi",
            feedback_layer_indices=[2, 2],
            feedback_gain_mode="shared",
        )
        invalid.append(values)
        values = small_onn_values()
        values.pop("feedback_layer_index")
        values.update(
            feedback_layer_mode="multi",
            feedback_layer_indices=[3, 2],
            feedback_gain_mode="shared",
        )
        invalid.append(values)
        values = small_onn_values()
        values.pop("feedback_layer_index")
        values.update(
            feedback_layer_mode="multi",
            feedback_layer_indices=[2, 4],
            feedback_gain_mode="shared",
        )
        invalid.append(values)
        values = small_onn_values()
        values.pop("feedback_layer_index")
        values.update(
            feedback_layer_mode="multi",
            feedback_layer_indices=[2, 3],
            feedback_gain_mode="invalid",
        )
        invalid.append(values)
        values = small_onn_values()
        values.pop("feedback_layer_index")
        values.update(
            feedback_layer_mode="multi",
            feedback_layer_indices=[2, 3],
            feedback_gain_mode="independent",
            feedback_gain_init=[1.0],
        )
        invalid.append(values)

        for config_values in invalid:
            with self.subTest(config_values=config_values):
                with self.assertRaises(ValueError):
                    ONNConfig.from_mapping(config_values)

    def test_multilayer_injection_selects_only_configured_layers_and_preserves_base_parameters(self):
        model = FeedbackFSONN(small_multi_onn_config("shared"))
        slot = torch.tensor([[[0.2, -0.3], [0.4, -0.1]]])
        feedback = torch.tensor([[[0.1, -0.2], [0.3, -0.4]]])
        before = [layer.phase_logits.detach().clone() for layer in model.slm_layers]

        _, debug = model._propagate_slot(slot, feedback=feedback)

        self.assertEqual(tuple(debug["feedback_phases"]), (2, 3, 4))
        self.assertEqual(debug["feedback_layer_indices"], (2, 3, 4))
        for layer_index in (2, 3, 4):
            self.assertIsNotNone(debug["feedback_phases"][layer_index])
        for old, layer in zip(before, model.slm_layers):
            self.assertTrue(torch.equal(old, layer.phase_logits))



if __name__ == "__main__":
    unittest.main()
