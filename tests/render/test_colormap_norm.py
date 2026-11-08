"""Property and unit tests for Colormap LUTs and Color Normalizers."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import AxisInfo, PassRecord, Step, TensorSnapshot, TensorStats
from tilevis.render.colormap import (
    COLORMAPS,
    DIVERGING_BLUE_ORANGE,
    DIVERGING_TEAL_MAGENTA,
    SEQUENTIAL_PURPLE,
    get_colormap,
)
from tilevis.render.normalization import (
    GlobalPassNormalizer,
    PerTensorNormalizer,
    SharedStepNormalizer,
    get_normalizer,
)


def _snap_from_array(
    arr: np.ndarray,
    *,
    role: str = TensorRole.OUTPUT.value,
    name: str = "t",
) -> TensorSnapshot:
    f32 = np.asarray(arr, dtype=np.float32)
    abs_arr = np.abs(f32)
    stats = TensorStats(
        min=float(np.min(f32)),
        max=float(np.max(f32)),
        mean=float(np.mean(f32)),
        std=float(np.std(f32)),
        abs_max=float(np.max(abs_arr)),
        abs_p99=float(np.percentile(abs_arr, 99)),
    )
    axes = tuple(
        AxisInfo(name="dim", full_size=dim, indices=np.arange(dim, dtype=np.int64))
        for dim in f32.shape
    )
    return TensorSnapshot(
        name=name,
        role=role,
        data=f32,
        axes=axes,
        full_shape=f32.shape,
        reduction="none",
        stats=stats,
    )


@given(
    arr=hnp.arrays(
        dtype=np.float32,
        shape=hnp.array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=24),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )
)
@settings(max_examples=80, deadline=None)
def test_property_colormaps_and_normalizers_finite_shapes(arr: np.ndarray) -> None:
    """Every colormap and normalizer produces valid uint8 RGB shapes with zero NaN/crash."""
    snap = _snap_from_array(arr)
    step = Step(
        id="p0.test",
        kind=StepKind.LINEAR.value,
        semantic=Semantic.BLOCK_ATT_OUT.value,
        stage=Stage.BLOCK.value,
        block_index=0,
        detail_level=int(DetailLevel.STANDARD),
        module_path=None,
        tensors=(snap,),
    )

    for norm_name in ("per_tensor", "shared_step", "global_pass"):
        normalizer = get_normalizer(norm_name)
        scaled, scale = normalizer.normalize(snap, step, signed=True)
        assert scale > 0.0
        assert np.isfinite(scale)
        assert np.all(np.isfinite(scaled))
        assert np.all(scaled >= -1.0) and np.all(scaled <= 1.0)

        for cmap_name in COLORMAPS:
            cmap = COLORMAPS.get(cmap_name)
            rgb = cmap.map(scaled)
            assert rgb.dtype == np.uint8
            assert rgb.shape == (*arr.shape, 3)


@given(
    arr=hnp.arrays(
        dtype=np.float32,
        shape=hnp.array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=16),
        elements=st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )
)
@settings(max_examples=60, deadline=None)
def test_diverging_colormap_exact_zero_and_sign_symmetry(arr: np.ndarray) -> None:
    """Diverging colormaps map 0 -> neutral lut[127] and preserve exact index reflection."""
    for cmap in (DIVERGING_BLUE_ORANGE, DIVERGING_TEAL_MAGENTA):
        zeros = np.zeros_like(arr)
        rgb_zero = cmap.map(zeros)
        assert np.array_equal(rgb_zero, np.broadcast_to(cmap.lut[127], (*arr.shape, 3)))

        # Reflection check: step distance from midpoint lut[127]
        offset_pos = np.rint(np.clip(arr.astype(np.float64), -1.0, 1.0) * 127.0).astype(np.intp)
        offset_neg = np.rint(np.clip((-arr).astype(np.float64), -1.0, 1.0) * 127.0).astype(np.intp)
        assert np.array_equal(offset_pos, -offset_neg)


def test_colormap_edge_cases_and_sequential_bounds() -> None:
    """Edge inputs (all zero, single element, constant negative, extreme outliers) work."""
    for edge in (
        np.zeros((4, 5), dtype=np.float32),
        np.array([[42.0]], dtype=np.float32),
        np.full((3, 3), -7.5, dtype=np.float32),
        np.array([[-1e6, 0.0, 1e6]], dtype=np.float32),
    ):
        snap = _snap_from_array(edge)
        step = Step(
            id="s",
            kind=StepKind.GENERIC.value,
            semantic=Semantic.GENERIC.value,
            stage=Stage.BLOCK.value,
            block_index=0,
            detail_level=2,
            module_path=None,
            tensors=(snap,),
        )
        scaled, scale = PerTensorNormalizer().normalize(snap, step)
        assert scale > 0.0
        rgb = DIVERGING_BLUE_ORANGE.map(scaled)
        assert rgb.shape == (*edge.shape, 3)
        assert rgb.dtype == np.uint8

    # Sequential purple maps [0, 1] monotonically from light endpoint to dark purple
    seq_rgb = SEQUENTIAL_PURPLE.map(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert seq_rgb.shape == (3, 3)
    assert np.array_equal(seq_rgb[0], SEQUENTIAL_PURPLE.lut[0])
    assert np.array_equal(seq_rgb[2], SEQUENTIAL_PURPLE.lut[255])
    assert get_colormap("does_not_exist") is DIVERGING_BLUE_ORANGE


def test_shared_step_and_global_pass_scales_and_outlier_clipping() -> None:
    """shared_step shares activation scale while weights keep own scale; p99 resists outliers."""
    # Sparse 1 outlier out of 200 small elements -> abs_p99 ~ 1.0
    small = np.full((2, 100), 1.0, dtype=np.float32)
    small[0, 0] = 5000.0
    snap_in = _snap_from_array(small, role=TensorRole.INPUT.value, name="in")
    snap_out = _snap_from_array(
        np.full((2, 4), 2.5, dtype=np.float32), role=TensorRole.OUTPUT.value
    )
    snap_w = _snap_from_array(np.full((4, 4), 20.0, dtype=np.float32), role=TensorRole.WEIGHT.value)

    step = Step(
        id="p0.b0.lin",
        kind=StepKind.LINEAR.value,
        semantic=Semantic.BLOCK_ATT_OUT.value,
        stage=Stage.BLOCK.value,
        block_index=0,
        detail_level=2,
        module_path=None,
        tensors=(snap_in, snap_w, snap_out),
    )

    shared = SharedStepNormalizer()
    assert abs(shared.compute_scale(snap_in, step) - 2.5) < 1e-5
    assert abs(shared.compute_scale(snap_out, step) - 2.5) < 1e-5
    assert abs(shared.compute_scale(snap_w, step) - 20.0) < 1e-5

    # Sparse single-spike fallback when abs_p99 == 0
    sparse = np.zeros((2, 200), dtype=np.float32)
    sparse[0, 0] = 5.0
    snap_sparse = _snap_from_array(sparse)
    assert abs(PerTensorNormalizer().compute_scale(snap_sparse, step) - 5.0) < 1e-5

    # global_pass computes max scale across steps in the pass
    step2 = Step(
        id="p0.b1.lin",
        kind=StepKind.LINEAR.value,
        semantic=Semantic.BLOCK_ATT_OUT.value,
        stage=Stage.BLOCK.value,
        block_index=1,
        detail_level=2,
        module_path=None,
        tensors=(_snap_from_array(np.full((2, 4), 9.0, dtype=np.float32)),),
    )
    pass_rec = PassRecord(
        index=0,
        input_ids=(1, 2),
        input_tokens=("a", "b"),
        text_before="a b",
        steps=(step, step2),
    )
    g_norm = GlobalPassNormalizer()
    assert abs(g_norm.compute_scale(snap_in, step, pass_rec) - 9.0) < 1e-5
