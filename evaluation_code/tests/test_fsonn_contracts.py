import unittest

import torch

from src.models.fsonn import (
    OpticalQKVConfig,
    TimeDivisionFSONN,
    merge_time_slots,
    split_time_slots,
)


class FSONNContractTests(unittest.TestCase):
    def test_time_slots_pad_and_restore_token_order(self):
        x = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)

        slots = split_time_slots(x, num_time_slots=3, token_chunk_size=2)

        self.assertEqual([tuple(slot.shape) for slot in slots], [(1, 2, 3)] * 3)
        self.assertTrue(torch.equal(slots[0], x[:, 0:2]))
        self.assertTrue(torch.equal(slots[1], x[:, 2:4]))
        self.assertTrue(torch.equal(slots[2][:, 0], x[:, 4]))
        self.assertTrue(torch.equal(slots[2][:, 1], torch.zeros(1, 3)))

        outputs = [torch.full((1, 2, 9), float(index)) for index in range(3)]
        restored = merge_time_slots(outputs, original_tokens=5)

        self.assertEqual(tuple(restored.shape), (1, 5, 9))
        self.assertTrue(torch.equal(restored[:, :2], outputs[0]))
        self.assertTrue(torch.equal(restored[:, 2:4], outputs[1]))
        self.assertTrue(torch.equal(restored[:, 4], outputs[2][:, 0]))

    def test_small_optical_network_has_finite_forward_and_parameter_gradients(self):
        config = OpticalQKVConfig(
            num_time_slots=3,
            token_chunk_size=2,
            input_dim=3,
            qkv_output_dim=9,
            grid_height=2,
            grid_width=9,
            pixel_pitch_um=8.0,
            wavelength_nm=532.0,
            num_slm_layers=3,
            input_to_first_slm_um=8.0,
            slm_intervals_um=(8.0, 8.0),
            last_slm_to_positive_detector_um=8.0,
            differential_detector_gap_um=2.0,
            asm_padding_factor=1.0,
        )
        model = TimeDivisionFSONN(config)
        x = torch.randn(1, 5, 3)

        output = model(x)
        loss = output.square().mean()
        loss.backward()

        self.assertEqual(tuple(output.shape), (1, 5, 9))
        self.assertTrue(torch.isfinite(output).all())
        trainable_grads = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_grads)
        self.assertTrue(all(grad is not None for grad in trainable_grads))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in trainable_grads))

    def test_attention_can_replace_only_qkv_and_capture_pre_dropout_projection(self):
        from src.models.utils.modules import Attention

        config = OpticalQKVConfig(
            num_time_slots=3,
            token_chunk_size=2,
            input_dim=2,
            qkv_output_dim=6,
            grid_height=2,
            grid_width=6,
            pixel_pitch_um=8.0,
            wavelength_nm=532.0,
            num_slm_layers=3,
            input_to_first_slm_um=8.0,
            slm_intervals_um=(8.0, 8.0),
            last_slm_to_positive_detector_um=8.0,
            differential_detector_gap_um=2.0,
            asm_padding_factor=1.0,
        )
        attention = Attention(
            dim=2,
            num_heads=1,
            use_sdpa=False,
            qkv_backend="fsonn_tdm",
            optical_config=config,
        )
        x = torch.randn(1, 5, 2)

        output, _, proj_output = attention(x, return_proj_output=True)

        self.assertEqual(tuple(output.shape), (1, 5, 2))
        self.assertEqual(tuple(proj_output.shape), (1, 5, 2))
        self.assertIsNotNone(attention.optical_qkv)

    def test_distillation_uses_attn_proj_output_and_freezes_electronic_paths(self):
        from src.models.optical_distillation import attention_proj_nmse, freeze_stage_one
        from src.models.utils.modules import Block

        config = OpticalQKVConfig(
            num_time_slots=3,
            token_chunk_size=2,
            input_dim=2,
            qkv_output_dim=6,
            grid_height=2,
            grid_width=6,
            pixel_pitch_um=8.0,
            wavelength_nm=532.0,
            num_slm_layers=3,
            input_to_first_slm_um=8.0,
            slm_intervals_um=(8.0, 8.0),
            last_slm_to_positive_detector_um=8.0,
            differential_detector_gap_um=2.0,
            asm_padding_factor=1.0,
        )
        teacher = Block(dim=2, num_heads=1, use_sdpa=False)
        student = Block(
            dim=2,
            num_heads=1,
            use_sdpa=False,
            qkv_backend="fsonn_tdm",
            optical_config=config,
        )
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        for parameter in student.parameters():
            parameter.requires_grad = "optical_qkv" in parameter.__class__.__name__.lower() or parameter.requires_grad

        freeze_stage_one(student)
        x = torch.randn(1, 5, 2)
        loss, teacher_proj, student_proj = attention_proj_nmse(teacher, student, x)
        loss.backward()

        self.assertEqual(tuple(teacher_proj.shape), (1, 5, 2))
        self.assertEqual(tuple(student_proj.shape), (1, 5, 2))
        self.assertFalse(teacher_proj.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            parameter.grad is not None
            for name, parameter in student.named_parameters()
            if "optical_qkv" in name
        ))
        self.assertIsNone(student.attn.proj.weight.grad)

    def test_installed_optical_qkv_follows_predictor_device(self):
        from src.models.predictor import VisionTransformerPredictor, install_optical_qkv

        config = OpticalQKVConfig(
            num_time_slots=3,
            token_chunk_size=2,
            input_dim=4,
            qkv_output_dim=12,
            grid_height=2,
            grid_width=12,
            pixel_pitch_um=8.0,
            wavelength_nm=532.0,
            num_slm_layers=3,
            input_to_first_slm_um=8.0,
            slm_intervals_um=(8.0, 8.0),
            last_slm_to_positive_detector_um=8.0,
            differential_detector_gap_um=2.0,
            asm_padding_factor=1.0,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = VisionTransformerPredictor(
            img_size=4,
            patch_size=2,
            num_frames=1,
            tubelet_size=1,
            embed_dim=4,
            predictor_embed_dim=4,
            depth=1,
            num_heads=1,
            use_mask_tokens=True,
            qkv_backend="electronic",
        ).to(device)

        install_optical_qkv(model, optical_config=config, replace_layers=[0])

        optical_devices = {
            parameter.device
            for name, parameter in model.named_parameters()
            if "optical_qkv" in name
        }
        self.assertEqual(optical_devices, {device})


    def test_predictor_keeps_unselected_blocks_electronic(self):
        from src.models.predictor import VisionTransformerPredictor

        config = OpticalQKVConfig(
            num_time_slots=3,
            token_chunk_size=3,
            input_dim=4,
            qkv_output_dim=12,
            grid_height=3,
            grid_width=12,
            pixel_pitch_um=8.0,
            wavelength_nm=532.0,
            num_slm_layers=3,
            input_to_first_slm_um=8.0,
            slm_intervals_um=(8.0, 8.0),
            last_slm_to_positive_detector_um=8.0,
            differential_detector_gap_um=2.0,
            asm_padding_factor=1.0,
        )
        model = VisionTransformerPredictor(
            img_size=4,
            patch_size=2,
            num_frames=1,
            tubelet_size=1,
            embed_dim=4,
            predictor_embed_dim=4,
            depth=2,
            num_heads=1,
            use_mask_tokens=True,
            qkv_backend="fsonn_tdm",
            optical_config=config,
            replace_layers=[0],
        )

        self.assertIsNone(model.predictor_blocks[0].attn.qkv)
        self.assertIsNotNone(model.predictor_blocks[0].attn.optical_qkv)
        self.assertIsNotNone(model.predictor_blocks[1].attn.qkv)
        self.assertIsNone(model.predictor_blocks[1].attn.optical_qkv)

    def test_optical_checkpoint_contains_only_optical_state_and_metadata(self):
        from src.models.optical_distillation import build_optical_checkpoint
        from src.models.utils.modules import Block

        config = OpticalQKVConfig(
            num_time_slots=3,
            token_chunk_size=2,
            input_dim=2,
            qkv_output_dim=6,
            grid_height=2,
            grid_width=6,
            pixel_pitch_um=8.0,
            wavelength_nm=532.0,
            num_slm_layers=3,
            input_to_first_slm_um=8.0,
            slm_intervals_um=(8.0, 8.0),
            last_slm_to_positive_detector_um=8.0,
            differential_detector_gap_um=2.0,
            asm_padding_factor=1.0,
        )
        student = Block(
            dim=2,
            num_heads=1,
            use_sdpa=False,
            qkv_backend="fsonn_tdm",
            optical_config=config,
        )
        checkpoint = build_optical_checkpoint(
            student,
            optical_config={"grid_height": 2},
            replace_layers=[0],
            teacher_checkpoint="teacher.pth",
            distill_target="attention_proj_output_pre_residual",
        )

        self.assertTrue(checkpoint["optical_state_dict"])
        self.assertTrue(all(
            "optical_qkv" in name
            for name in checkpoint["optical_state_dict"]
        ))
        self.assertNotIn("attn.qkv.weight", checkpoint["optical_state_dict"])
        self.assertEqual(checkpoint["replace_layers"], [0])
        self.assertEqual(checkpoint["teacher_checkpoint"], "teacher.pth")


if __name__ == "__main__":
    unittest.main()
