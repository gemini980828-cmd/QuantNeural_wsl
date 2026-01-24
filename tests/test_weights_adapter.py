"""
Tests for src/weights_adapter.py

Covers:
- Softmax row sums to one and non-negative
- Softmax temperature validation
- Rank row sums to one and non-negative
- TopK selects exactly k and equal weights
- Invalid method raises
- NaN or inf in scores raises
- Non-monotonic index raises
- Duplicate index raises
- Duplicate columns raises
- Max weight enforced and row sums to one
- Infeasible max weight raises
- Determinism repeat call identical
"""

import numpy as np
import pandas as pd
import pytest

from src.weights_adapter import scores_to_target_weights


class TestSoftmax:
    """Test softmax method."""
    
    def test_softmax_row_sums_to_one_and_nonnegative(self):
        """Test softmax produces non-negative weights summing to 1."""
        scores = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 1.0, 0.0],
            "C": [0.5, 0.5, 0.5],
        }, index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))
        
        weights = scores_to_target_weights(scores, method="softmax")
        
        # All weights non-negative
        assert np.all(weights.values >= 0)
        
        # Each row sums to 1
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-12)
    
    def test_softmax_higher_score_higher_weight(self):
        """Test higher score gets higher weight."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [5.0],
            "C": [0.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights = scores_to_target_weights(scores, method="softmax")
        
        # A > B > C
        assert weights.loc["2023-01-01", "A"] > weights.loc["2023-01-01", "B"]
        assert weights.loc["2023-01-01", "B"] > weights.loc["2023-01-01", "C"]
    
    def test_softmax_temperature_validation(self):
        """Test temperature <= 0 raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="temperature"):
            scores_to_target_weights(scores, method="softmax", temperature=0.0)
        
        with pytest.raises(ValueError, match="temperature"):
            scores_to_target_weights(scores, method="softmax", temperature=-1.0)
    
    def test_softmax_low_temperature_sharpens(self):
        """Test low temperature makes distribution sharper."""
        scores = pd.DataFrame({
            "A": [2.0],
            "B": [1.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights_t1 = scores_to_target_weights(scores, method="softmax", temperature=1.0)
        weights_t01 = scores_to_target_weights(scores, method="softmax", temperature=0.1)
        
        # Lower temperature -> winner takes more
        assert weights_t01.loc["2023-01-01", "A"] > weights_t1.loc["2023-01-01", "A"]


class TestSoftmaxScoreTransform:
    """Test optional pre-softmax score transforms."""
    
    def test_softmax_winsorize_zscore_reduces_outlier_concentration(self):
        """Winsorize+zscore should prevent near-100% allocation from an extreme outlier."""
        scores = pd.DataFrame({
            "A": [100.0],
            "B": [1.0],
            "C": [1.0],
            "D": [1.0],
            "E": [1.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        w_plain = scores_to_target_weights(scores, method="softmax", temperature=1.0)
        w_tx = scores_to_target_weights(
            scores,
            method="softmax",
            temperature=1.0,
            score_transform="winsorize_zscore",
            winsorize_q_low=0.0,
            winsorize_q_high=0.75,
        )
        
        max_plain = float(w_plain.max(axis=1).iloc[0])
        max_tx = float(w_tx.max(axis=1).iloc[0])
        
        assert max_plain > 0.99
        assert max_tx < 0.50
        
        np.testing.assert_allclose(w_tx.sum(axis=1).values, 1.0, atol=1e-12)
    
    def test_invalid_score_transform_raises(self):
        """Invalid score_transform should raise ValueError."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="score_transform"):
            scores_to_target_weights(scores, method="softmax", score_transform="nope")
    
    def test_score_transform_requires_softmax(self):
        """score_transform is only supported for softmax to avoid surprising rank/topk behavior."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="method='softmax'"):
            scores_to_target_weights(scores, method="rank", score_transform="zscore")


class TestSoftmaxTopK:
    """Test softmax_topk method (top-k prefilter then softmax)."""
    
    def test_softmax_topk_sparsifies_and_sums_to_one(self):
        """softmax_topk should assign 0 to non-selected and sum weights to 1."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [5.0],
            "C": [0.0],
            "D": [-1.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        w = scores_to_target_weights(scores, method="softmax_topk", top_k=2, temperature=1.0)
        
        # Only A and B are active
        assert w.loc["2023-01-01", "A"] > 0
        assert w.loc["2023-01-01", "B"] > 0
        assert w.loc["2023-01-01", "C"] == 0.0
        assert w.loc["2023-01-01", "D"] == 0.0
        
        np.testing.assert_allclose(w.sum(axis=1).values, 1.0, atol=1e-12)
        assert np.all(w.values >= 0.0)
    
    def test_softmax_topk_deterministic_tie_breaking(self):
        """On ties, selection is by column name ascending (deterministic)."""
        scores = pd.DataFrame({
            "C": [1.0],
            "A": [1.0],
            "B": [1.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        w = scores_to_target_weights(scores, method="softmax_topk", top_k=2, temperature=1.0)
        
        # Selected should be A and B (alphabetical), equal softmax weights
        assert w.loc["2023-01-01", "A"] > 0
        assert w.loc["2023-01-01", "B"] > 0
        assert w.loc["2023-01-01", "C"] == 0.0
        np.testing.assert_allclose(w.loc["2023-01-01", "A"], 0.5, atol=1e-12)
        np.testing.assert_allclose(w.loc["2023-01-01", "B"], 0.5, atol=1e-12)
    
    def test_softmax_topk_max_weight_preserves_sparsity(self):
        """max_weight should cap within active set and keep non-selected at 0."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [0.0],
            "C": [-5.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        w = scores_to_target_weights(
            scores,
            method="softmax_topk",
            top_k=2,
            temperature=1.0,
            max_weight=0.60,
        )
        
        assert w.loc["2023-01-01", "A"] <= 0.60 + 1e-12
        assert w.loc["2023-01-01", "B"] >= 0.40 - 1e-12
        assert w.loc["2023-01-01", "C"] == 0.0
        np.testing.assert_allclose(w.sum(axis=1).values, 1.0, atol=1e-12)
    
    def test_softmax_topk_infeasible_max_weight_raises(self):
        """Infeasible max_weight for the active set should raise."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="Infeasible"):
            scores_to_target_weights(scores, method="softmax_topk", top_k=2, max_weight=0.40)


class TestRank:
    """Test rank method."""
    
    def test_rank_row_sums_to_one_and_nonnegative(self):
        """Test rank produces non-negative weights summing to 1."""
        scores = pd.DataFrame({
            "A": [1.0, 3.0, 2.0],
            "B": [3.0, 1.0, 2.0],
            "C": [2.0, 2.0, 1.0],
        }, index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))
        
        weights = scores_to_target_weights(scores, method="rank")
        
        # All weights non-negative
        assert np.all(weights.values >= 0)
        
        # Each row sums to 1
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-12)
    
    def test_rank_higher_score_higher_weight(self):
        """Test higher score gets higher weight."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [5.0],
            "C": [0.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights = scores_to_target_weights(scores, method="rank")
        
        # A > B > C
        assert weights.loc["2023-01-01", "A"] > weights.loc["2023-01-01", "B"]
        assert weights.loc["2023-01-01", "B"] > weights.loc["2023-01-01", "C"]


class TestRankTopK:
    """Test rank method with top_k sparsification (Task 9.1.0)."""
    
    def test_rank_topk_sparsifies_correctly(self):
        """Test rank + top_k selects only top_k assets with rank weights."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [5.0],
            "C": [0.0],
            "D": [7.0],
            "E": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights = scores_to_target_weights(scores, method="rank", top_k=2)
        
        # Only A (10) and D (7) should be selected
        assert weights.loc["2023-01-01", "A"] > 0  # Highest
        assert weights.loc["2023-01-01", "D"] > 0  # Second highest
        assert weights.loc["2023-01-01", "B"] == 0.0
        assert weights.loc["2023-01-01", "C"] == 0.0
        assert weights.loc["2023-01-01", "E"] == 0.0
        
        # Row sums to 1
        np.testing.assert_allclose(weights.sum(axis=1).values, 1.0, atol=1e-12)
        
        # A gets higher weight (rank=2) than D (rank=1)
        # Weights: A=2/3, D=1/3
        np.testing.assert_allclose(weights.loc["2023-01-01", "A"], 2/3, atol=1e-12)
        np.testing.assert_allclose(weights.loc["2023-01-01", "D"], 1/3, atol=1e-12)
    
    def test_rank_topk_deterministic_tie_breaking(self):
        """Test rank + top_k has deterministic tie-breaking by column name."""
        # All equal scores, shuffled column order
        scores = pd.DataFrame({
            "C": [1.0],
            "A": [1.0],
            "B": [1.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights = scores_to_target_weights(scores, method="rank", top_k=2)
        
        # Tie-breaking: alphabetical order -> A and B selected, C excluded
        assert weights.loc["2023-01-01", "A"] > 0  # Selected (rank 2)
        assert weights.loc["2023-01-01", "B"] > 0  # Selected (rank 1)
        assert weights.loc["2023-01-01", "C"] == 0.0  # Excluded
        
        # A gets higher weight (first alphabetically among ties)
        # A: rank=2, B: rank=1 -> A=2/3, B=1/3
        np.testing.assert_allclose(weights.loc["2023-01-01", "A"], 2/3, atol=1e-12)
        np.testing.assert_allclose(weights.loc["2023-01-01", "B"], 1/3, atol=1e-12)
    
    def test_rank_topk_out_of_range_raises(self):
        """Test rank + top_k raises on invalid top_k."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="top_k"):
            scores_to_target_weights(scores, method="rank", top_k=0)
        
        with pytest.raises(ValueError, match="top_k"):
            scores_to_target_weights(scores, method="rank", top_k=4)


class TestMaxWeightSparsity:
    """Test max_weight preserves sparsity (Task 9.1.0)."""
    
    def test_max_weight_preserves_sparsity_no_leakage(self):
        """Test max_weight does not leak weight into zero-weight assets."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [5.0],
            "C": [0.0],  # Will be excluded by top_k=2
        }, index=pd.to_datetime(["2023-01-01"]))
        
        # top_k=2 -> only A, B active
        # max_weight=0.6 is feasible (0.6*2=1.2 > 1.0)
        weights = scores_to_target_weights(scores, method="topk", top_k=2, max_weight=0.6)
        
        # C must remain 0
        assert weights.loc["2023-01-01", "C"] == 0.0
        
        # A and B must be <= 0.6
        assert weights.loc["2023-01-01", "A"] <= 0.6 + 1e-12
        assert weights.loc["2023-01-01", "B"] <= 0.6 + 1e-12
        
        # Row sums to 1
        np.testing.assert_allclose(weights.sum(axis=1).values, 1.0, atol=1e-12)
    
    def test_rank_topk_max_weight_preserves_sparsity(self):
        """Test rank + top_k + max_weight preserves sparsity."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [5.0],
            "C": [3.0],
            "D": [1.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        # top_k=3 -> A, B, C active, D excluded
        # max_weight=0.45 is feasible (0.45*3=1.35 > 1.0)
        weights = scores_to_target_weights(scores, method="rank", top_k=3, max_weight=0.45)
        
        # D must remain 0
        assert weights.loc["2023-01-01", "D"] == 0.0
        
        # A, B, C must be <= 0.45
        for col in ["A", "B", "C"]:
            assert weights.loc["2023-01-01", col] > 0
            assert weights.loc["2023-01-01", col] <= 0.45 + 1e-12
        
        # Row sums to 1
        np.testing.assert_allclose(weights.sum(axis=1).values, 1.0, atol=1e-12)
    
    def test_infeasible_max_weight_on_sparse_active_set_raises(self):
        """Test infeasible max_weight for sparse methods raises."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        # topk=2, max_weight=0.4 -> 0.4*2=0.8 < 1.0 -> infeasible
        with pytest.raises(ValueError, match="Infeasible"):
            scores_to_target_weights(scores, method="topk", top_k=2, max_weight=0.4)
        
        # rank + top_k=1, max_weight=0.9 -> 0.9*1=0.9 < 1.0 -> infeasible
        with pytest.raises(ValueError, match="Infeasible"):
            scores_to_target_weights(scores, method="rank", top_k=1, max_weight=0.9)


class TestTopK:
    """Test topk method."""
    
    def test_topk_selects_exactly_k_and_equal_weights(self):
        """Test topk selects exactly k assets with equal weights."""
        scores = pd.DataFrame({
            "A": [10.0],
            "B": [5.0],
            "C": [0.0],
            "D": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights = scores_to_target_weights(scores, method="topk", top_k=2)
        
        # Top 2 are A and B
        assert weights.loc["2023-01-01", "A"] == 0.5
        assert weights.loc["2023-01-01", "B"] == 0.5
        assert weights.loc["2023-01-01", "C"] == 0.0
        assert weights.loc["2023-01-01", "D"] == 0.0
    
    def test_topk_requires_top_k_parameter(self):
        """Test topk raises when top_k is not provided."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="top_k is required"):
            scores_to_target_weights(scores, method="topk")
    
    def test_topk_k_out_of_range_raises(self):
        """Test top_k out of range raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="top_k"):
            scores_to_target_weights(scores, method="topk", top_k=0)
        
        with pytest.raises(ValueError, match="top_k"):
            scores_to_target_weights(scores, method="topk", top_k=4)
    
    def test_topk_deterministic_tie_breaking(self):
        """Test topk has deterministic tie-breaking."""
        # All same score -> deterministic selection by column name
        scores = pd.DataFrame({
            "C": [1.0],
            "A": [1.0],
            "B": [1.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights1 = scores_to_target_weights(scores, method="topk", top_k=2)
        weights2 = scores_to_target_weights(scores, method="topk", top_k=2)
        
        pd.testing.assert_frame_equal(weights1, weights2)


class TestInvalidInputs:
    """Test fail-fast validation."""
    
    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="method"):
            scores_to_target_weights(scores, method="invalid")
    
    def test_nan_in_scores_raises(self):
        """Test NaN in scores raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0, np.nan],
            "B": [2.0, 3.0],
        }, index=pd.to_datetime(["2023-01-01", "2023-02-01"]))
        
        with pytest.raises(ValueError, match="finite"):
            scores_to_target_weights(scores)
    
    def test_inf_in_scores_raises(self):
        """Test inf in scores raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0, np.inf],
            "B": [2.0, 3.0],
        }, index=pd.to_datetime(["2023-01-01", "2023-02-01"]))
        
        with pytest.raises(ValueError, match="finite"):
            scores_to_target_weights(scores)
    
    def test_non_monotonic_index_raises(self):
        """Test non-monotonic index raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 3.0, 4.0],
        }, index=pd.to_datetime(["2023-03-01", "2023-01-01", "2023-02-01"]))
        
        with pytest.raises(ValueError, match="monotonic"):
            scores_to_target_weights(scores)
    
    def test_duplicate_index_raises(self):
        """Test duplicate index raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0, 2.0],
            "B": [2.0, 3.0],
        }, index=pd.to_datetime(["2023-01-01", "2023-01-01"]))  # Duplicate!
        
        with pytest.raises(ValueError, match="unique"):
            scores_to_target_weights(scores)
    
    def test_duplicate_columns_raises(self):
        """Test duplicate columns raises ValueError."""
        scores = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.to_datetime(["2023-01-01", "2023-02-01"]),
            columns=["A", "A"]  # Duplicate!
        )
        
        with pytest.raises(ValueError, match="unique"):
            scores_to_target_weights(scores)
    
    def test_scores_must_be_dataframe(self):
        """Test numpy array raises ValueError."""
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with pytest.raises(ValueError, match="DataFrame"):
            scores_to_target_weights(scores)
    
    def test_less_than_2_assets_raises(self):
        """Test k_assets < 2 raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0, 2.0],
        }, index=pd.to_datetime(["2023-01-01", "2023-02-01"]))
        
        with pytest.raises(ValueError, match="k_assets >= 2"):
            scores_to_target_weights(scores)


class TestMaxWeight:
    """Test max_weight enforcement."""
    
    def test_max_weight_enforced_and_row_sums_to_one(self):
        """Test max_weight caps weights and row still sums to 1."""
        scores = pd.DataFrame({
            "A": [10.0],  # Would dominate without cap
            "B": [1.0],
            "C": [0.1],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        weights = scores_to_target_weights(scores, method="softmax", max_weight=0.5)
        
        # All weights <= max_weight
        assert np.all(weights.values <= 0.5 + 1e-12)
        
        # Row sums to 1
        np.testing.assert_allclose(weights.sum(axis=1).values, 1.0, atol=1e-12)
    
    def test_infeasible_max_weight_raises(self):
        """Test max_weight too low for k assets raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        # max_weight=0.2 with k=3: 0.2*3=0.6 < 1.0 -> infeasible
        with pytest.raises(ValueError, match="Infeasible"):
            scores_to_target_weights(scores, max_weight=0.2)
    
    def test_max_weight_at_boundary(self):
        """Test max_weight at exact feasibility boundary works."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
            "C": [3.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        # max_weight=1/3 with k=3: exactly feasible
        weights = scores_to_target_weights(scores, method="rank", max_weight=1/3)
        
        # All equal at max
        np.testing.assert_allclose(weights.values, 1/3, atol=1e-10)
    
    def test_max_weight_out_of_range_raises(self):
        """Test max_weight out of (0, 1] raises ValueError."""
        scores = pd.DataFrame({
            "A": [1.0],
            "B": [2.0],
        }, index=pd.to_datetime(["2023-01-01"]))
        
        with pytest.raises(ValueError, match="max_weight"):
            scores_to_target_weights(scores, max_weight=0.0)
        
        with pytest.raises(ValueError, match="max_weight"):
            scores_to_target_weights(scores, max_weight=1.5)


class TestDeterminism:
    """Test determinism."""
    
    def test_determinism_repeat_call_identical(self):
        """Test repeated calls produce identical results."""
        scores = pd.DataFrame({
            "A": [1.5, 2.5, 3.5],
            "B": [3.0, 1.0, 2.0],
            "C": [2.0, 3.0, 1.0],
        }, index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))
        
        # Softmax
        w1 = scores_to_target_weights(scores, method="softmax", temperature=0.5)
        w2 = scores_to_target_weights(scores, method="softmax", temperature=0.5)
        pd.testing.assert_frame_equal(w1, w2)
        
        # Rank
        w3 = scores_to_target_weights(scores, method="rank")
        w4 = scores_to_target_weights(scores, method="rank")
        pd.testing.assert_frame_equal(w3, w4)
        
        # TopK
        w5 = scores_to_target_weights(scores, method="topk", top_k=2)
        w6 = scores_to_target_weights(scores, method="topk", top_k=2)
        pd.testing.assert_frame_equal(w5, w6)


class TestOutputShape:
    """Test output shape and type."""
    
    def test_output_same_shape_index_columns(self):
        """Test output has same shape, index, columns as input."""
        dates = pd.to_datetime(["2023-01-01", "2023-02-01"])
        cols = ["AAPL", "MSFT", "GOOG"]
        
        scores = pd.DataFrame({
            "AAPL": [1.0, 2.0],
            "MSFT": [2.0, 1.0],
            "GOOG": [1.5, 1.5],
        }, index=dates)
        
        weights = scores_to_target_weights(scores, method="softmax")
        
        assert weights.shape == scores.shape
        assert list(weights.index) == list(scores.index)
        assert list(weights.columns) == list(scores.columns)
