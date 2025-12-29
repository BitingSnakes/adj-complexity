import pytest
import numpy as np
from adjusted_complexity import AdjustedComplexityCalculator


class TestAdjustedComplexityManualTrace:
    """
    Unit tests validating the AdjustedComplexityCalculator against
    manual traces of the algorithm definitions.
    """

    def setup_method(self):
        self.calc = AdjustedComplexityCalculator()

    def test_unconditional_trace_balanced_vector(self):
        """
        MANUAL TRACE:
        Input: [1, 1, 0, 0]
        L = 4, w = 2
        p = 0.5 -> Entropy = 1.0
        Baseline = L * H = 4 * 1.0 = 4.0

        K_eff calculation:
        - Binomial(4, 2) = 6
        - log2(6) = 2.5849625
        - Overhead = 2 * log2(4) = 4.0
        - K_eff = 6.5849625

        Expected R_eff = 6.5849625 / 4.0 = 1.64624
        Expected D_eff = 4.0 - 6.5849625 = -2.5849625
        """
        X = np.array([[1, 1, 0, 0]])

        r_eff, d_eff = self.calc.compute_unconditional_batch(X)

        # Calculations constants
        expected_log_binom = np.log2(6)
        expected_overhead = 4.0
        expected_k = expected_log_binom + expected_overhead
        expected_baseline = 4.0

        expected_r = expected_k / expected_baseline
        expected_d = expected_baseline - expected_k

        # Assertions
        assert r_eff[0] == pytest.approx(expected_r, abs=1e-5), (
            f"Expected Ratio {expected_r}, got {r_eff[0]}"
        )

        assert d_eff[0] == pytest.approx(expected_d, abs=1e-5), (
            f"Expected Deficiency {expected_d}, got {d_eff[0]}"
        )

    def test_unconditional_trace_constant_vector(self):
        """
        MANUAL TRACE (Singularity):
        Input: [1, 1, 1, 1]
        L = 4, w = 4
        p = 1.0 -> Entropy = 0.0
        Baseline = 0.0

        Per algorithm definition:
        If Baseline == 0, R_eff = INFINITY, D_eff = 0
        """
        X = np.array([[1, 1, 1, 1]])

        r_eff, d_eff = self.calc.compute_unconditional_batch(X)

        assert r_eff[0] == np.inf, "Constant vector should result in Infinite Ratio"
        assert d_eff[0] == 0.0, "Constant vector should have 0 deficiency"

    def test_conditional_trace_uncorrelated(self):
        """
        MANUAL TRACE:
        X = [1, 1, 0, 0]
        Y = [1, 0, 1, 0]

        Joint Counts:
        (x,y):
        idx 0: 1,1 -> Count 11: 1
        idx 1: 1,0 -> Count 10: 1
        idx 2: 0,1 -> Count 01: 1
        idx 3: 0,0 -> Count 00: 1

        Condition Y=0: Count=2. X values in this slice: [1, 0]. w=1.
        Condition Y=1: Count=2. X values in this slice: [1, 0]. w=1.

        Costs:
        Seg Y=0: log2(2C1) = log2(2) = 1.0
        Seg Y=1: log2(2C1) = log2(2) = 1.0
        Overhead: 2 * log2(4) = 4.0
        Total K = 1.0 + 1.0 + 4.0 = 6.0

        Baseline:
        H(X|Y=0) = H(0.5) = 1.0
        H(X|Y=1) = H(0.5) = 1.0
        Total Baseline = 4 * ((0.5*1) + (0.5*1)) = 4.0

        Expected R = 6.0 / 4.0 = 1.5
        """
        X = np.array([[1, 1, 0, 0]])
        Y = np.array([[1, 0, 1, 0]])

        r_cond = self.calc.compute_conditional_batch(X, Y)

        expected_r = 6.0 / 4.0

        assert r_cond[0] == pytest.approx(expected_r, abs=1e-5), (
            f"Expected Conditional Ratio {expected_r}, got {r_cond[0]}"
        )

    def test_conditional_trace_perfect_copy(self):
        """
        MANUAL TRACE:
        X = [0, 1, 0, 1]
        Y = [0, 1, 0, 1]

        Condition Y=0: X is always 0. k=0, n=2. log2(2C0) = 0.
        Condition Y=1: X is always 1. k=2, n=2. log2(2C2) = 0.

        Total K = 0 + 0 + Overhead(4) = 4.0

        Baseline:
        H(X|Y) = 0 (Perfectly determined)

        Ratio: 4.0 / 0.0 -> Infinity
        """
        X = np.array([[0, 1, 0, 1]])
        Y = np.array([[0, 1, 0, 1]])

        r_cond = self.calc.compute_conditional_batch(X, Y)

        assert r_cond[0] == np.inf, (
            "Perfectly determined conditional should be Infinite"
        )

    def test_batch_consistency(self):
        """
        Verify that processing a batch yields the same results as processing
        individually.
        """
        x1 = [1, 1, 0, 0]
        x2 = [1, 1, 1, 1]

        X_batch = np.array([x1, x2])

        r_batch, d_batch = self.calc.compute_unconditional_batch(X_batch)

        # Run individually
        r_1, d_1 = self.calc.compute_unconditional_batch(np.array([x1]))
        r_2, d_2 = self.calc.compute_unconditional_batch(np.array([x2]))

        assert r_batch[0] == pytest.approx(r_1[0])
        assert r_batch[1] == r_2[0]  # r_2 is inf
        assert d_batch[0] == pytest.approx(d_1[0])
        assert d_batch[1] == d_2[0]
