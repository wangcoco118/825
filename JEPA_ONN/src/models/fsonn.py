"""Differentiable single-wavelength free-space optical QKV projection."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


_TRANSFER_CACHE: Dict[Tuple, torch.Tensor] = {}


@dataclass(frozen=True)
class ONNConfig:
    """Single-detector serial ONN configuration for the feedback Predictor."""

    input_dim: int = 384
    output_dim: int = 384
    num_slm_layers: int = 4
    chunk_tokens: int = 196
    grid_height: int = 196
    grid_width: int = 384
    feedback_mode: str = "fixed_middle_phase"
    feedback_enabled: bool = True
    feedback_layer_mode: Optional[str] = None
    feedback_layer_index: Optional[int] = None
    feedback_layer_indices: Optional[Tuple[int, ...]] = None
    feedback_gain_mode: Optional[str] = None
    feedback_phase_max_rad: float = 1.5707963267948966
    feedback_gain_init: Union[float, Tuple[float, ...]] = 1.0
    feedback_gain_epsilon: float = 1.0e-6
    readout_mode: str = "intensity_minus_learnable_offset"
    input_encoding_mode: str = "signed_phase"
    pixel_pitch_um: float = 8.0
    wavelength_nm: float = 532.0
    slm_intervals_um: Tuple[float, ...] = (25000.0, 25000.0, 25000.0)
    input_to_first_slm_um: float = 50000.0
    last_slm_to_detector_um: float = 50000.0
    asm_padding_factor: float = 2.0
    eps: float = 1e-6
    learnable_intensity_offset: bool = True
    use_differential_detector: bool = False

    @classmethod
    def from_mapping(cls, values):
        values = dict(values or {})
        if "feedback_layer_index" in values and "feedback_layer_indices" in values:
            raise ValueError(
                "feedback_layer_index and feedback_layer_indices are mutually exclusive"
            )
        forbidden = {
            "positive_field",
            "negative_field",
            "positive_intensity",
            "negative_intensity",
            "positive_gain_raw",
            "negative_gain_raw",
            "differential_detector_gap_um",
            "detector_split_ratio",
            "optical_qkv",
            "qkv_backend",
            "qkv_output_dim",
            "replace_layers",
            "distill_target",
            "attention_nmse_weight",
            "add_original_qkv_bias",
        }
        present = sorted(forbidden.intersection(values))
        if present:
            raise ValueError(
                "onn config contains disabled legacy/differential fields: "
                + ", ".join(present)
            )
        allowed = set(cls.__dataclass_fields__)
        normalized = {key: value for key, value in values.items() if key in allowed}
        if "slm_intervals_um" in normalized:
            normalized["slm_intervals_um"] = tuple(normalized["slm_intervals_um"])
        if "feedback_layer_indices" in normalized:
            normalized["feedback_layer_indices"] = tuple(
                int(index) for index in normalized["feedback_layer_indices"]
            )
        gain_init = normalized.get("feedback_gain_init")
        if isinstance(gain_init, (list, tuple)):
            normalized["feedback_gain_init"] = tuple(float(value) for value in gain_init)
        return cls(**normalized)

    def __post_init__(self):
        if self.feedback_mode == "fixed_middle":
            object.__setattr__(self, "feedback_mode", "fixed_middle_phase")
        positive = (
            self.input_dim,
            self.output_dim,
            self.num_slm_layers,
            self.chunk_tokens,
            self.grid_height,
            self.grid_width,
            self.pixel_pitch_um,
            self.wavelength_nm,
            self.input_to_first_slm_um,
            self.last_slm_to_detector_um,
            self.asm_padding_factor,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("ONN dimensions, distances, and sampling values must be positive")
        if self.input_dim != self.output_dim:
            raise ValueError("single-detector ONN requires input_dim == output_dim")
        if self.grid_height != self.chunk_tokens or self.grid_width != self.output_dim:
            raise ValueError("ONN grid must be [chunk_tokens, output_dim]")
        if self.feedback_mode != "fixed_middle_phase":
            raise ValueError("only feedback_mode='fixed_middle_phase' is supported")
        if self.feedback_phase_max_rad <= 0:
            raise ValueError("feedback_phase_max_rad must be positive")
        if self.feedback_gain_epsilon <= 0:
            raise ValueError("feedback_gain_epsilon must be positive")
        if self.readout_mode != "intensity_minus_learnable_offset":
            raise ValueError("only intensity_minus_learnable_offset is supported")
        if self.input_encoding_mode != "signed_phase":
            raise ValueError("only signed_phase input encoding is supported")

        layer_mode = self.feedback_layer_mode
        if layer_mode is None:
            if self.feedback_layer_indices is not None:
                raise ValueError(
                    "feedback_layer_indices requires feedback_layer_mode='multi'"
                )
            layer_mode = "single"
            object.__setattr__(self, "feedback_layer_mode", layer_mode)
            if self.feedback_layer_index is None:
                object.__setattr__(self, "feedback_layer_index", 2)
        if layer_mode not in {"single", "multi"}:
            raise ValueError("feedback_layer_mode must be 'single' or 'multi'")

        if layer_mode == "single":
            if self.feedback_layer_index is None:
                raise ValueError("single feedback requires feedback_layer_index")
            if self.feedback_layer_indices is not None:
                raise ValueError(
                    "single feedback cannot define feedback_layer_indices"
                )
            if self.feedback_gain_mode is not None:
                raise ValueError(
                    "single feedback cannot define feedback_gain_mode"
                )
            if not 0 <= int(self.feedback_layer_index) < self.num_slm_layers:
                raise ValueError(
                    "feedback_layer_index must identify an existing SLM layer"
                )
        else:
            if self.feedback_layer_index is not None:
                raise ValueError(
                    "multi feedback cannot define feedback_layer_index"
                )
            indices = self.feedback_layer_indices
            if indices is None or len(indices) < 2:
                raise ValueError(
                    "multi feedback requires at least two feedback_layer_indices"
                )
            if tuple(sorted(set(indices))) != tuple(indices):
                raise ValueError(
                    "feedback_layer_indices must be unique and strictly increasing"
                )
            if any(not 0 <= int(index) < self.num_slm_layers for index in indices):
                raise ValueError(
                    "feedback_layer_indices must identify existing SLM layers"
                )
            if self.feedback_gain_mode not in {"shared", "independent"}:
                raise ValueError(
                    "multi feedback_gain_mode must be 'shared' or 'independent'"
                )

        gain_values = self.feedback_gain_initial_values()
        if any(value <= self.feedback_gain_epsilon for value in gain_values):
            raise ValueError(
                "all effective feedback_gain_init values must exceed "
                "feedback_gain_epsilon"
            )
        if len(self.slm_intervals_um) != self.num_slm_layers - 1:
            raise ValueError("slm_intervals_um must contain one distance between each pair of SLMs")
        if any(distance <= 0 for distance in self.slm_intervals_um):
            raise ValueError("all SLM interval distances must be positive")
        if self.use_differential_detector:
            raise ValueError("differential detector is disabled for the feedback ONN")

    @property
    def active_feedback_layer_indices(self) -> Tuple[int, ...]:
        if self.feedback_layer_mode == "single":
            return (int(self.feedback_layer_index),)
        return tuple(int(index) for index in self.feedback_layer_indices)

    def feedback_gain_initial_values(self) -> Tuple[float, ...]:
        gain_init = self.feedback_gain_init
        is_sequence = isinstance(gain_init, tuple)
        if self.feedback_layer_mode in {None, "single"}:
            if is_sequence:
                raise ValueError("single feedback_gain_init must be a scalar")
            return (float(gain_init),)
        if self.feedback_gain_mode == "shared":
            if is_sequence:
                raise ValueError("multi/shared feedback_gain_init must be a scalar")
            return (float(gain_init),)
        if is_sequence:
            if len(gain_init) != len(self.feedback_layer_indices):
                raise ValueError(
                    "multi/independent feedback_gain_init length must match "
                    "feedback_layer_indices"
                )
            return tuple(float(value) for value in gain_init)
        return tuple(float(gain_init) for _ in self.feedback_layer_indices)


class FeedbackFSONN(nn.Module):
    """Serial phase-only ONN with one intensity detector and signed input encoding."""

    def __init__(self, config: ONNConfig):
        super().__init__()
        self.config = config
        self.slm_layers = nn.ModuleList(
            [
                PhaseSLM(config.grid_height, config.grid_width)
                for _ in range(config.num_slm_layers)
            ]
        )
        self.input_scale_raw = nn.Parameter(torch.zeros(()))
        self.feedback_norm = nn.LayerNorm(
            config.output_dim,
            elementwise_affine=False,
        )
        initial_gains = torch.tensor(
            config.feedback_gain_initial_values(),
            dtype=torch.float32,
        )
        initial_softplus = initial_gains - config.feedback_gain_epsilon
        feedback_gain_raw = initial_softplus + torch.log(
            -torch.expm1(-initial_softplus)
        )
        if config.feedback_layer_mode == "single" or config.feedback_gain_mode == "shared":
            feedback_gain_raw = feedback_gain_raw.squeeze(0)
        self.feedback_gain_raw = nn.Parameter(feedback_gain_raw)
        self.intensity_offset = (
            nn.Parameter(torch.zeros(1, 1, config.output_dim))
            if config.learnable_intensity_offset
            else None
        )

    def _positive_parameter(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw) + self.config.eps

    def _effective_feedback_gains(self) -> torch.Tensor:
        return F.softplus(self.feedback_gain_raw) + self.config.feedback_gain_epsilon

    def _feedback_to_phases(
        self,
        feedback: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        if not self.config.feedback_enabled:
            return {}
        normalized = self.feedback_norm(feedback)
        gains = self._effective_feedback_gains()
        indices = self.config.active_feedback_layer_indices
        if self.config.feedback_layer_mode == "single" or self.config.feedback_gain_mode == "shared":
            shared_phase = self.config.feedback_phase_max_rad * torch.tanh(
                gains * normalized
            )
            return {index: shared_phase for index in indices}
        return {
            index: self.config.feedback_phase_max_rad * torch.tanh(
                gains[position] * normalized
            )
            for position, index in enumerate(indices)
        }

    def _feedback_to_phase(self, feedback: torch.Tensor) -> torch.Tensor:
        phases = self._feedback_to_phases(feedback)
        if len(phases) != 1:
            raise ValueError("_feedback_to_phase is only valid for single feedback")
        return next(iter(phases.values()))

    def _resolve_feedback_layer_indices(
        self,
        feedback_layer_indices: Optional[Sequence[int]],
        feedback_layer_index: Optional[int],
    ) -> Tuple[int, ...]:
        if feedback_layer_indices is not None and feedback_layer_index is not None:
            raise ValueError(
                "feedback_layer_index and feedback_layer_indices are mutually exclusive"
            )
        configured = self.config.active_feedback_layer_indices
        if feedback_layer_indices is not None:
            requested = tuple(int(index) for index in feedback_layer_indices)
        elif feedback_layer_index is not None:
            requested = (int(feedback_layer_index),)
        else:
            requested = configured
        if requested != configured:
            raise ValueError(
                "runtime feedback layer selection must match ONNConfig"
            )
        return configured

    def _encode(self, slot: torch.Tensor) -> torch.Tensor:
        scale = self._positive_parameter(self.input_scale_raw)
        normalized = torch.clamp(slot / scale, min=-1.0, max=1.0)
        phase = torch.where(
            normalized >= 0,
            torch.zeros_like(normalized),
            torch.full_like(normalized, torch.pi),
        )
        encoded = torch.polar(normalized.abs(), phase)
        if self.config.grid_width > self.config.input_dim:
            zeros = torch.zeros(
                slot.shape[0],
                slot.shape[1],
                self.config.grid_width - self.config.input_dim,
                device=slot.device,
                dtype=encoded.dtype,
            )
            encoded = torch.cat([encoded, zeros], dim=-1)
        return encoded

    def _propagate_slot(
        self,
        slot: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
        feedback_layer_index: Optional[int] = None,
        feedback_layer_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        field = self._encode(slot)
        field = band_limited_angular_spectrum(
            field,
            self.config.input_to_first_slm_um,
            self.config.pixel_pitch_um,
            self.config.wavelength_nm,
            self.config.asm_padding_factor,
        )
        selected_indices = self._resolve_feedback_layer_indices(
            feedback_layer_indices,
            feedback_layer_index,
        )
        feedback_phases = {}
        if feedback is not None:
            if feedback.shape != slot.shape:
                raise ValueError("feedback must match the current slot shape")
            feedback_phases = self._feedback_to_phases(feedback)
        feedback_phase = (
            next(iter(feedback_phases.values()))
            if len(feedback_phases) == 1
            else None
        )
        phase_abs_max = (
            torch.stack(
                [phase.abs().max() for phase in feedback_phases.values()]
            ).max()
            if feedback_phases
            else torch.zeros((), device=slot.device, dtype=slot.dtype)
        )
        debug = {
            "feedback_gain": self._effective_feedback_gains(),
            "feedback_phase": feedback_phase,
            "feedback_phases": feedback_phases,
            "feedback_phase_abs_max": phase_abs_max,
            "feedback_layer_index": (
                selected_indices[0]
                if self.config.feedback_layer_mode == "single"
                else None
            ),
            "feedback_layer_indices": selected_indices,
            "feedback_layer_mode": self.config.feedback_layer_mode,
            "feedback_gain_mode": self.config.feedback_gain_mode,
            "feedback_used": feedback is not None and bool(feedback_phases),
        }
        for layer_index, slm in enumerate(self.slm_layers):
            field = slm(field, phase_delta=feedback_phases.get(layer_index))
            debug[f"slm_{layer_index + 1}_field"] = field
            if layer_index < len(self.slm_layers) - 1:
                field = band_limited_angular_spectrum(
                    field,
                    self.config.slm_intervals_um[layer_index],
                    self.config.pixel_pitch_um,
                    self.config.wavelength_nm,
                    self.config.asm_padding_factor,
                )

        field = band_limited_angular_spectrum(
            field,
            self.config.last_slm_to_detector_um,
            self.config.pixel_pitch_um,
            self.config.wavelength_nm,
            self.config.asm_padding_factor,
        )
        intensity = field.abs().square()
        output = intensity
        if self.intensity_offset is not None:
            output = output - self.intensity_offset
        debug["intensity"] = intensity
        debug["output"] = output
        return output, debug

    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False,
        feedback: Optional[torch.Tensor] = None,
        feedback_layer_index: Optional[int] = None,
        feedback_layer_indices: Optional[Sequence[int]] = None,
    ):
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected input tensor with shape [B, N, {self.config.input_dim}]"
            )
        slots = split_time_slots(x, 1, self.config.chunk_tokens)
        outputs = []
        debug_slots = []
        for slot in slots:
            output, debug = self._propagate_slot(
                slot,
                feedback=feedback,
                feedback_layer_index=feedback_layer_index,
                feedback_layer_indices=feedback_layer_indices,
            )
            outputs.append(output)
            debug_slots.append(debug)
        merged = merge_time_slots(outputs, original_tokens=x.shape[1])
        if return_debug:
            return merged, {"slots": debug_slots, "merged": merged}
        return merged


@dataclass(frozen=True)
class OpticalQKVConfig:
    num_time_slots: int = 3
    token_chunk_size: int = 523
    input_dim: int = 384
    qkv_output_dim: int = 1152
    grid_height: int = 523
    grid_width: int = 1152
    pixel_pitch_um: float = 8.0
    wavelength_nm: float = 532.0
    num_slm_layers: int = 3
    slm_intervals_um: Tuple[float, ...] = (50000.0, 50000.0)
    input_to_first_slm_um: float = 50000.0
    last_slm_to_positive_detector_um: float = 50000.0
    differential_detector_gap_um: float = 5000.0
    detector_split_ratio: float = 0.5
    asm_padding_factor: float = 2.0
    eps: float = 1e-6
    learnable_intensity_offset: bool = False

    @classmethod
    def from_mapping(cls, values):
        if values.get("propagation_method", "band_limited_asm") != "band_limited_asm":
            raise ValueError("only propagation_method=band_limited_asm is supported")
        if values.get("slm_modulation_mode", "phase") != "phase":
            raise ValueError("only phase SLM modulation is supported")
        if values.get("input_encoding_mode", "signed_phase") != "signed_phase":
            raise ValueError("only signed_phase input encoding is supported")
        if values.get("add_original_qkv_bias", False):
            raise ValueError("optical QKV cannot add the original electronic bias")
        if values.get("readout_additive_bias", False):
            raise ValueError("differential readout cannot use additive bias")
        if values.get("noise_enabled", False):
            raise ValueError("noise is disabled in the first optical implementation")
        allowed = set(cls.__dataclass_fields__)
        normalized = {key: value for key, value in values.items() if key in allowed}
        if "slm_intervals_um" in normalized:
            normalized["slm_intervals_um"] = tuple(normalized["slm_intervals_um"])
        return cls(**normalized)

    def __post_init__(self):
        positive = (
            self.token_chunk_size,
            self.input_dim,
            self.qkv_output_dim,
            self.grid_height,
            self.grid_width,
            self.pixel_pitch_um,
            self.wavelength_nm,
            self.num_slm_layers,
            self.input_to_first_slm_um,
            self.last_slm_to_positive_detector_um,
            self.differential_detector_gap_um,
            self.asm_padding_factor,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("optical dimensions, distances, and sampling values must be positive")
        if self.num_slm_layers < 1:
            raise ValueError("num_slm_layers must be positive")
        if self.qkv_output_dim not in (self.input_dim, 3 * self.input_dim):
            raise ValueError(
                "qkv_output_dim must equal input_dim for a feature ONN "
                "or 3 * input_dim for QKV"
            )
        if self.grid_height != self.token_chunk_size:
            raise ValueError("grid_height must equal token_chunk_size")
        if self.grid_width != self.qkv_output_dim:
            raise ValueError("grid_width must equal qkv_output_dim")
        if self.input_dim > self.grid_width:
            raise ValueError("input_dim cannot exceed grid_width")
        if len(self.slm_intervals_um) != self.num_slm_layers - 1:
            raise ValueError("slm_intervals_um must contain one distance between each pair of SLMs")
        if any(distance <= 0 for distance in self.slm_intervals_um):
            raise ValueError("all SLM interval distances must be positive")
        if not 0.0 < self.detector_split_ratio < 1.0:
            raise ValueError("detector_split_ratio must be strictly between zero and one")
        if self.num_time_slots * self.token_chunk_size <= 0:
            raise ValueError("time-slot capacity must be positive")


def split_time_slots(
    x: torch.Tensor,
    num_time_slots: int,
    token_chunk_size: int,
) -> List[torch.Tensor]:
    if x.ndim != 3:
        raise ValueError("expected token tensor with shape [B, N, C]")
    if num_time_slots <= 0 or token_chunk_size <= 0:
        raise ValueError("num_time_slots and token_chunk_size must be positive")
    capacity = num_time_slots * token_chunk_size
    if x.shape[1] > capacity:
        raise ValueError(f"input has {x.shape[1]} tokens but capacity is {capacity}")
    padding = capacity - x.shape[1]
    if padding:
        x = F.pad(x, (0, 0, 0, padding))
    return list(x.split(token_chunk_size, dim=1))


def merge_time_slots(
    outputs: Sequence[torch.Tensor],
    original_tokens: int,
) -> torch.Tensor:
    if not outputs:
        raise ValueError("at least one time-slot output is required")
    if original_tokens <= 0:
        raise ValueError("original_tokens must be positive")
    merged = torch.cat(list(outputs), dim=1)
    if original_tokens > merged.shape[1]:
        raise ValueError("original_tokens exceeds merged token count")
    return merged[:, :original_tokens, :]


def _transfer_function(
    height: int,
    width: int,
    pixel_pitch_um: float,
    wavelength_nm: float,
    distance_um: float,
    device: torch.device,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    key = (
        device.type,
        device.index,
        complex_dtype,
        height,
        width,
        float(pixel_pitch_um),
        float(wavelength_nm),
        float(distance_um),
    )
    cached = _TRANSFER_CACHE.get(key)
    if cached is not None:
        return cached

    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    pixel_pitch_m = float(pixel_pitch_um) * 1e-6
    wavelength_m = float(wavelength_nm) * 1e-9
    distance_m = float(distance_um) * 1e-6

    fy = torch.fft.fftfreq(height, d=pixel_pitch_m, device=device, dtype=real_dtype)
    fx = torch.fft.fftfreq(width, d=pixel_pitch_m, device=device, dtype=real_dtype)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    k = 2.0 * torch.pi / wavelength_m
    kx = 2.0 * torch.pi * fx_grid
    ky = 2.0 * torch.pi * fy_grid
    kz_squared = k * k - kx.square() - ky.square()
    propagating = kz_squared > 0
    kz = torch.sqrt(torch.clamp(kz_squared, min=0.0))

    # Matsushima-style rectangular band limit for the sampled angular spectrum.
    extent_x = width * pixel_pitch_m
    extent_y = height * pixel_pitch_m
    limit_x = 1.0 / (wavelength_m * (1.0 + (2.0 * abs(distance_m) / extent_x) ** 2) ** 0.5)
    limit_y = 1.0 / (wavelength_m * (1.0 + (2.0 * abs(distance_m) / extent_y) ** 2) ** 0.5)
    band_limited = (fx_grid.abs() <= limit_x) & (fy_grid.abs() <= limit_y)

    phase = kz * distance_m
    transfer = torch.polar(
        torch.ones_like(phase),
        phase,
    ) * propagating.to(real_dtype) * band_limited.to(real_dtype)
    transfer = transfer.to(complex_dtype)
    _TRANSFER_CACHE[key] = transfer.detach()
    return transfer


def band_limited_angular_spectrum(
    field: torch.Tensor,
    distance_um: float,
    pixel_pitch_um: float,
    wavelength_nm: float,
    padding_factor: float = 2.0,
) -> torch.Tensor:
    if not torch.is_complex(field):
        raise TypeError("BL-ASM expects a complex-valued field")
    if field.ndim < 2:
        raise ValueError("field must have height and width dimensions")
    if distance_um <= 0:
        raise ValueError("distance_um must be positive")
    if padding_factor < 1.0:
        raise ValueError("padding_factor must be at least one")

    height, width = field.shape[-2:]
    padded_height = max(height, int(round(height * padding_factor)))
    padded_width = max(width, int(round(width * padding_factor)))
    pad_height = padded_height - height
    pad_width = padded_width - width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    padded = F.pad(field, (left, right, top, bottom))
    transfer = _transfer_function(
        padded_height,
        padded_width,
        pixel_pitch_um,
        wavelength_nm,
        distance_um,
        field.device,
        field.dtype,
    )
    propagated = torch.fft.ifft2(torch.fft.fft2(padded) * transfer)
    return propagated[..., top : top + height, left : left + width]


class PhaseSLM(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.phase_logits = nn.Parameter(torch.zeros(height, width))

    def forward(
        self,
        field: torch.Tensor,
        phase_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_phase = 2.0 * torch.pi * torch.sigmoid(self.phase_logits)
        total_phase = base_phase
        if phase_delta is not None:
            total_phase = total_phase + phase_delta
        return field * torch.polar(
            torch.ones_like(total_phase),
            total_phase,
        ).to(field.dtype)


class TimeDivisionFSONN(nn.Module):
    def __init__(self, config: OpticalQKVConfig):
        super().__init__()
        self.config = config
        self.slm_layers = nn.ModuleList(
            [PhaseSLM(config.grid_height, config.grid_width) for _ in range(config.num_slm_layers)]
        )
        self.input_scale_raw = nn.Parameter(torch.zeros(()))
        self.positive_gain_raw = nn.Parameter(torch.zeros(1, 1, config.qkv_output_dim))
        self.negative_gain_raw = nn.Parameter(torch.zeros(1, 1, config.qkv_output_dim))
        self.intensity_offset = (
            nn.Parameter(torch.zeros(1, 1, config.qkv_output_dim))
            if config.learnable_intensity_offset
            else None
        )

    def _positive_parameter(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw) + self.config.eps

    def _encode(self, slot: torch.Tensor) -> torch.Tensor:
        scale = self._positive_parameter(self.input_scale_raw)
        normalized = torch.clamp(slot / scale, min=-1.0, max=1.0)
        phase = torch.where(
            normalized >= 0,
            torch.zeros_like(normalized),
            torch.full_like(normalized, torch.pi),
        )
        encoded = torch.polar(normalized.abs(), phase)
        zeros = torch.zeros(
            slot.shape[0],
            slot.shape[1],
            self.config.grid_width - self.config.input_dim,
            device=slot.device,
            dtype=encoded.dtype,
        )
        return torch.cat([encoded, zeros], dim=-1)

    def _propagate_slot(
        self,
        slot: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
        feedback_layer_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        field = self._encode(slot)
        debug = {"encoded_field": field}
        field = band_limited_angular_spectrum(
            field,
            self.config.input_to_first_slm_um,
            self.config.pixel_pitch_um,
            self.config.wavelength_nm,
            self.config.asm_padding_factor,
        )
        if feedback is not None:
            if feedback.shape != slot.shape:
                raise ValueError(
                    "feedback must match the current slot shape, got "
                    f"{tuple(feedback.shape)} and {tuple(slot.shape)}"
                )
            if feedback_layer_index is None:
                feedback_layer_index = len(self.slm_layers) // 2
            if not 0 <= int(feedback_layer_index) < len(self.slm_layers):
                raise ValueError(
                    "feedback_layer_index must identify an existing SLM layer"
                )

        for index, slm in enumerate(self.slm_layers):
            if feedback is not None and index == int(feedback_layer_index):
                field = field + self._encode(feedback)
            field = slm(field)
            debug[f"slm_{index + 1}_field"] = field
            if index < len(self.slm_layers) - 1:
                field = band_limited_angular_spectrum(
                    field,
                    self.config.slm_intervals_um[index],
                    self.config.pixel_pitch_um,
                    self.config.wavelength_nm,
                    self.config.asm_padding_factor,
                )

        split = float(self.config.detector_split_ratio) ** 0.5
        positive_field = split * field
        negative_field = (1.0 - float(self.config.detector_split_ratio)) ** 0.5 * field
        positive_field = band_limited_angular_spectrum(
            positive_field,
            self.config.last_slm_to_positive_detector_um,
            self.config.pixel_pitch_um,
            self.config.wavelength_nm,
            self.config.asm_padding_factor,
        )
        negative_field = band_limited_angular_spectrum(
            negative_field,
            self.config.last_slm_to_positive_detector_um + self.config.differential_detector_gap_um,
            self.config.pixel_pitch_um,
            self.config.wavelength_nm,
            self.config.asm_padding_factor,
        )
        positive_intensity = positive_field.abs().square()
        negative_intensity = negative_field.abs().square()
        positive_gain = self._positive_parameter(self.positive_gain_raw)
        negative_gain = self._positive_parameter(self.negative_gain_raw)
        output = positive_gain * positive_intensity - negative_gain * negative_intensity
        if self.intensity_offset is not None:
            output = output - self.intensity_offset
        debug["positive_intensity"] = positive_intensity
        debug["negative_intensity"] = negative_intensity
        debug["output"] = output
        return output, debug

    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False,
        feedback: Optional[torch.Tensor] = None,
        feedback_layer_index: Optional[int] = None,
    ):
        if x.ndim != 3:
            raise ValueError("expected input tensor with shape [B, N, C]")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected input_dim={self.config.input_dim}, got {x.shape[-1]}")
        if feedback is not None and self.config.num_time_slots != 1:
            raise ValueError("feedback is only supported for one-slot ONN steps")

        slots = split_time_slots(
            x,
            self.config.num_time_slots,
            self.config.token_chunk_size,
        )
        outputs = []
        debug_slots = []
        for slot in slots:
            output, debug = self._propagate_slot(
                slot,
                feedback=feedback,
                feedback_layer_index=feedback_layer_index,
            )
            outputs.append(output)
            debug_slots.append(debug)
        merged = merge_time_slots(outputs, original_tokens=x.shape[1])
        if return_debug:
            return merged, {"slots": debug_slots, "merged": merged}
        return merged
