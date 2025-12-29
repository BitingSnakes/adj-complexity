from typing import Tuple
from scipy.special import gammaln as lgamma

import numpy as np


class AdjustedComplexityCalculator:
    """
    Implements Adjusted Kolmogorov Complexity algorithms using
    Combinatorial Shell Coding surrogates.

    Ref: Section 6 (Combinatorial Shell Coding) and Section 8 (Effective Surrogates).
    """

    def __init__(self):
        """
        Initializes the calculator.

        CONSTANTS:
        - LN2: Natural logarithm of 2, used for base conversion.
        """
        self.LN2 = np.log(2)

    def compute_unconditional_batch(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algorithm 1: Unconditional Adjusted Complexity (Batch Processing)

        Computes the empirical entropy, effective combinatorial complexity,
        and the adjusted ratio R_eff for a batch of binary words.

        Args:
            X (np.ndarray): Binary Matrix of size N x L (N words, length L).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - R_eff (Vector of size N): Adjusted Ratio.
                - D_eff (Vector of size N): Deficiency.
        """
        # 1. Initialization and Dimensions
        # DEFINE N = rows(X), L = cols(X)
        N, L = X.shape

        # 2. Compute Empirical Weights (Vectorized)
        # Equation: w = Sum(X, axis=1)
        w = np.sum(X, axis=1)

        # 3. Compute Empirical Probability
        # Equation: p = w / L
        p = w / L

        # 4. Compute Empirical Entropy H(nu_L)
        # Equation: H_emp = -p log2(p) - (1-p) log2(1-p)
        # Note: We use a helper to safely handle 0 log 0 = 0
        H_emp = self._batch_binary_entropy(p)

        # 5. Compute Combinatorial Baseline (Shell Size)
        # Equation (Denominator): n * H(nu_n) -> L * H_emp
        shell_baseline = L * H_emp

        # 6. Compute Effective Complexity K_eff (Combinatorial Shell Code)
        # From Section 6: K_comb(x) = log2(Binomial(L, w)) + O(log L)
        # We use lgamma for numerical stability:
        # log2(nCk) = (lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)) / ln(2)

        # Vectorized Log-Binomial calculation
        log_nCk = (lgamma(L + 1) - lgamma(w + 1) - lgamma(L - w + 1)) / self.LN2

        # Add overhead term representing description of (n, k).
        # Approx 2 * log2(L) (standard prefix code cost).
        overhead = 2 * np.log2(L)

        K_eff = log_nCk + overhead

        # 7. Compute Ratios and Deficiency
        # Initialize output arrays
        R_eff = np.zeros(N)
        D_eff = np.zeros(N)

        # Handle division by zero where H_emp is 0 (constant words)
        non_constant_mask = H_emp > 0

        # Case: H_emp > 0
        # Eq: R_eff(x) = K_eff(x) / (L * H(nu_L))
        R_eff[non_constant_mask] = (
            K_eff[non_constant_mask] / shell_baseline[non_constant_mask]
        )

        # Eq: d_eff(x) = L * H(nu_L) - K_eff(x)
        D_eff[non_constant_mask] = (
            shell_baseline[non_constant_mask] - K_eff[non_constant_mask]
        )

        # Case: H_emp == 0 (Constant words)
        # Convention: Set to Infinity as per Section 4 discussions on singularities
        R_eff[~non_constant_mask] = np.inf
        D_eff[~non_constant_mask] = (
            0.0  # No deficiency for perfectly compressible constant strings
        )

        return R_eff, D_eff

    def compute_conditional_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Algorithm 2: Conditional Adjusted Complexity

        Calculates R(x|y) using empirical conditional entropy and conditional shell coding.

        Args:
            X (np.ndarray): Target Binary Matrix (N x L).
            Y (np.ndarray): Condition Binary Matrix (N x L).

        Returns:
            np.ndarray: R_cond (Vector of size N).
        """
        N, L = X.shape
        R_cond = np.zeros(N)

        # Loop implementation as per Algorithm 2 pseudocode
        for i in range(N):
            # 1. Extract row vectors
            x_vec = X[i, :]
            y_vec = Y[i, :]

            # 2. Compute Joint Frequencies (Contingency Table)
            # Count occurrences of (0,0), (0,1), (1,0), (1,1) where pairs are (x, y)
            # Note: The pseudocode notation implies x corresponds to rows in table?
            # Let's align strictly with pseudocode: count_00 = count(x=0 AND y=0)

            # Create boolean masks for clarity
            x_0 = x_vec == 0
            x_1 = x_vec == 1
            y_0 = y_vec == 0
            y_1 = y_vec == 1

            count_00 = np.sum(x_0 & y_0)  # x=0, y=0
            count_01 = np.sum(x_0 & y_1)  # x=0, y=1
            count_10 = np.sum(x_1 & y_0)  # x=1, y=0
            count_11 = np.sum(x_1 & y_1)  # x=1, y=1

            # 3. Compute Marginals for Y (Side Information)
            count_y0 = count_00 + count_10
            count_y1 = count_01 + count_11

            # 4. Compute Conditional Entropy H_emp(X|Y)
            # Eq: H(X|Y) = P(Y=0)H(X|Y=0) + P(Y=1)H(X|Y=1)

            term_y0 = 0.0
            if count_y0 > 0:
                # p(x=1 | y=0)
                p_x1_given_y0 = count_10 / count_y0
                # Weighted entropy: H(p) * P(Y=0)
                term_y0 = self._scalar_binary_entropy(p_x1_given_y0) * (count_y0 / L)

            term_y1 = 0.0
            if count_y1 > 0:
                # p(x=1 | y=1)
                p_x1_given_y1 = count_11 / count_y1
                # Weighted entropy: H(p) * P(Y=1)
                term_y1 = self._scalar_binary_entropy(p_x1_given_y1) * (count_y1 / L)

            H_cond = term_y0 + term_y1
            conditional_baseline = L * H_cond

            # 5. Compute Conditional Effective Complexity K_eff(x|y)
            # Cost = log2(Binomial(count_y0, count_10)) + log2(Binomial(count_y1, count_11))

            cost_segment_0 = self._log_binomial(count_y0, count_10)
            cost_segment_1 = self._log_binomial(count_y1, count_11)

            # Overhead to describe the counts (approx 2 * log(L))
            overhead = 2 * np.log2(L)

            K_cond = cost_segment_0 + cost_segment_1 + overhead

            # 6. Compute Ratio
            if conditional_baseline == 0:
                # Fully determined by side information or constant relative to Y
                R_cond[i] = np.inf
            else:
                R_cond[i] = K_cond / conditional_baseline

        return R_cond

    def _batch_binary_entropy(self, p: np.ndarray) -> np.ndarray:
        """
        Helper: Computes binary entropy element-wise for a vector.
        Formula: -p log2(p) - (1-p) log2(1-p)
        Handles singularities at p=0 and p=1.
        """
        # Initialize with zeros
        H = np.zeros_like(p)

        # Mask for values strictly between 0 and 1
        mask = (p > 0) & (p < 1)

        # Compute entropy only for valid probabilities
        p_valid = p[mask]
        H[mask] = -1 * (
            p_valid * np.log2(p_valid) + (1 - p_valid) * np.log2(1 - p_valid)
        )

        return H

    def _scalar_binary_entropy(self, p: float) -> float:
        """
        Helper: Computes binary entropy for a scalar float.
        """
        if p <= 0 or p >= 1:
            return 0.0
        return -1 * (p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def _log_binomial(self, n: int, k: int) -> float:
        """
        Helper: Computes log2(nCk) using lgamma for stability.
        Eq: log2(nCk) = (lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)) / ln(2)
        """
        if k < 0 or k > n:
            return -np.inf  # Impossible event
        if k == 0 or k == n:
            return 0.0  # log2(1) = 0

        return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / self.LN2


# --- Test Block (Isolated execution) ---
if __name__ == "__main__":
    # Create instance
    calc = AdjustedComplexityCalculator()

    print("--- Test 1: Unconditional Batch ---")
    # Example: 3 words of length 10
    # Word 1: Random-ish
    # Word 2: Low entropy (mostly zeros)
    # Word 3: Constant (all ones)
    X_test = np.array(
        [
            [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # Balanced
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Low weight
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Constant
        ]
    )

    R_eff, D_eff = calc.compute_unconditional_batch(X_test)

    print(f"Input Shape: {X_test.shape}")
    print(f"R_eff: {R_eff}")
    print(f"D_eff: {D_eff}")

    print("\n--- Test 2: Conditional Batch ---")
    # X is target, Y is side info
    # Case 1: X equals Y (Perfect prediction)
    # Case 2: X is inverse of Y (Perfect prediction)
    # Case 3: Uncorrelated
    X_cond = np.array(
        [[1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0]]
    )
    Y_cond = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],  # Same as X[0]
            [0, 0, 0, 0, 1, 1, 1, 1],  # Inverse of X[1]
            [0, 0, 0, 0, 0, 0, 0, 0],  # Uncorrelated/Constant
        ]
    )

    R_cond = calc.compute_conditional_batch(X_cond, Y_cond)
    print(f"R_cond: {R_cond}")
