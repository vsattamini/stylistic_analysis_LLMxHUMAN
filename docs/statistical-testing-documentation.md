# Statistical Testing Documentation

## Overview

This document describes all statistical tests and machine learning analyses performed in the study. The methodology combines non-parametric hypothesis testing for univariate feature analysis with both parametric and non-parametric classifiers for multivariate prediction.

**Implementation:** `src/tests.py`, `src/models.py`, `src/fuzzy.py`, `EDA.ipynb`

**Libraries:** scipy.stats, scikit-learn

---

## 1. Univariate Analysis: Feature Comparison

### 1.1 Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

**Purpose:** Compare distributions of each stylometric feature between human-written and LLM-generated texts.

**Why Non-Parametric?**

Non-parametric tests were chosen for the following empirical and theoretical reasons:

1. **Violation of normality assumption:** Visual inspection of feature distributions (histograms, Q-Q plots) revealed significant deviations from normality, including skewness and heavy tails.
2. **Presence of outliers:** Many features exhibit extreme values that would disproportionately influence parametric tests.
3. **Robustness:** Non-parametric tests make no assumptions about the underlying distribution, relying only on rank ordering.
4. **Statistical power:** For non-normal data, non-parametric tests often have higher statistical power than their parametric alternatives.

**Hypotheses:**
- **H₀ (null hypothesis):** The two populations (human and LLM) have identical distributions for the feature
- **H₁ (alternative hypothesis):** The two populations have different distributions (two-sided test)

**Test Statistic:**

The Mann-Whitney U statistic is based on the sum of ranks. For two samples X and Y:

```
U = R₁ - n₁(n₁ + 1)/2
```

where:
- R₁ = sum of ranks for sample 1
- n₁ = size of sample 1

**Implementation:**

```python
from scipy.stats import mannwhitneyu

human_values = features_df[features_df['label'] == 'human']['ttr']
llm_values = features_df[features_df['label'] == 'llm']['ttr']

statistic, p_value = mannwhitneyu(human_values, llm_values,
                                  alternative='two-sided')
```

**Significance Level:** α = 0.05

**Decision Rule:** Reject H₀ if p < 0.05 (after FDR correction)

**Citation:** Mann & Whitney (1947), Wilcoxon (1945)

---

### 1.2 Effect Size: Cliff's Delta (δ)

**Purpose:** Quantify the **practical significance** (magnitude) of differences between groups.

**Rationale:** Statistical significance (p-value) indicates whether an effect exists, but does not measure its magnitude. With large samples, even trivial differences become statistically significant. Effect sizes provide interpretable measures of practical importance.

**Why Cliff's Delta?**

Cliff's delta is the appropriate effect size measure for non-parametric comparisons:

1. **Distribution-free:** Makes no assumptions about normality
2. **Robust to outliers:** Based on ordinal comparisons, not means
3. **Intuitive interpretation:** Represents the probability that a randomly selected value from one group exceeds a randomly selected value from the other group
4. **Paired with Mann-Whitney U:** Cliff's delta is the non-parametric analogue to Cohen's d

**Formula:**

```
δ = (D⁺ - D⁻) / (n₁ × n₂)
```

where:
- D⁺ = number of pairs where x_i > y_j (human > LLM)
- D⁻ = number of pairs where x_i < y_j (human < LLM)
- n₁ = sample size of human texts
- n₂ = sample size of LLM texts

**Range:** δ ∈ [-1, +1]

**Interpretation:**
- δ = +1: All human values > all LLM values (complete separation)
- δ = 0: Distributions completely overlap (50% probability either way)
- δ = -1: All human values < all LLM values (complete reverse separation)

**Effect Size Thresholds (Romano et al., 2006):**

| |δ| Range    | Interpretation | Meaning |
|--------------|----------------|---------|
| < 0.147      | Negligible     | Trivial effect, distributions largely overlap |
| 0.147-0.330  | Small          | Noticeable but modest difference |
| 0.330-0.474  | Medium         | Moderate difference, clear separation |
| ≥ 0.474      | Large          | Substantial difference, strong separation |

**Implementation:**

```python
def cliffs_delta(x, y):
    """Calculate Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    # Use broadcasting for efficient computation
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    delta = (greater - less) / float(n1 * n2)
    return delta
```

**Citation:** Cliff (1993), Romano et al. (2006)

---

### 1.3 Multiple Comparison Correction: False Discovery Rate (FDR)

**Problem:** When testing multiple hypotheses simultaneously, the probability of making at least one Type I error (false positive) increases dramatically.

**Family-Wise Error Rate (FWER):**

For m independent tests at significance level α:

```
P(at least 1 false positive) = 1 - (1 - α)^m
```

For our study with 10 features at α = 0.05:

```
FWER ≈ 1 - (1 - 0.05)^10 ≈ 0.40 (40% chance of false discovery)
```

**Solution: Benjamini-Hochberg FDR Correction**

The False Discovery Rate (FDR) is the expected proportion of false discoveries among all rejected hypotheses. The Benjamini-Hochberg procedure controls FDR at level α.

**Procedure:**

1. Perform all m tests, obtaining p-values: p₁, p₂, ..., p_m
2. Sort p-values in ascending order: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍m₎
3. For each test i, calculate the critical value:
   ```
   α_i = (i/m) × α
   ```
4. Find the largest index k where p₍k₎ ≤ α_k
5. Reject H₀ for all tests i = 1, 2, ..., k

**Adjusted p-values (q-values):**

For each test i, the q-value is computed as:

```
q_i = min_{j≥i} { m × p₍j₎ / j }
```

This is the minimum FDR at which the test would be rejected.

**Advantage over Bonferroni:**

- **Bonferroni correction:** α_bonf = α/m (very conservative, low power)
- **FDR correction:** Adaptive threshold, higher power while controlling expected false discovery proportion

For m = 10 tests:
- Bonferroni threshold: 0.05/10 = 0.005 (very stringent)
- FDR thresholds: 0.005, 0.010, 0.015, ..., 0.050 (adaptive)

**FDR Level:** α = 0.05 (controls expected proportion of false discoveries at 5%)

**Implementation:**

```python
from scipy.stats import false_discovery_control

p_values = [...]  # List of 10 p-values from Mann-Whitney tests
reject, q_values = false_discovery_control(p_values, alpha=0.05)
```

Alternative manual implementation:

```python
def fdr_bh(p_values):
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Compute q-values
    q = np.empty(m, dtype=float)
    min_coeff = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        coeff = m / rank * sorted_p[i]
        min_coeff = min(min_coeff, coeff)
        q[i] = min_coeff

    # Reorder to original order and cap at 1
    q_values = np.minimum(1.0, q[np.argsort(sorted_indices)])
    return q_values.tolist()
```

**Citation:** Benjamini & Hochberg (1995)

---

## 2. Multivariate Analysis: Dimensionality Reduction and Visualization

### 2.1 Principal Component Analysis (PCA)

**Purpose:**
- Reduce high-dimensional feature space (10 features) to 2D for visualization
- Understand which features contribute most to variance
- Identify latent patterns in feature space

**Method:** Principal Component Analysis (unsupervised)

**Mathematical Foundation:**

PCA finds orthogonal directions (principal components) that maximize variance:

1. **Standardization:** Center and scale features to zero mean, unit variance
   ```
   X_scaled = (X - μ) / σ
   ```

2. **Covariance matrix:** Compute C = (1/n) X^T X

3. **Eigendecomposition:** Solve Cv = λv for eigenvalues λ and eigenvectors v

4. **Projection:** Transform data onto top k eigenvectors
   ```
   Z = X × V_k
   ```
   where V_k contains the k eigenvectors with largest eigenvalues

**Components Retained:** k = 2 (for 2D visualization)

**Variance Explained:**

Report the cumulative proportion of variance captured:
```
explained_variance_ratio = [λ₁/(Σλ), λ₂/(Σλ)]
```

**Note:** PCA is **unsupervised** and does not use class labels during fitting. It finds directions of maximum variance regardless of class membership.

**Purpose in This Study:**

1. **Exploratory visualization:** Visualize the 10-dimensional feature space in 2D
2. **Class separation assessment:** Observe whether human and LLM texts cluster separately
3. **Feature interpretation:** Examine principal component loadings to understand which features define the "LLM-ness" axis

**Principal Component Loadings:**

Each PC is a linear combination of original features:
```
PC1 = w₁₁×sent_mean + w₁₂×sent_std + ... + w₁₁₀×bigram_repeat_ratio
```

The weights (loadings) indicate each feature's contribution to the component.

**Implementation:**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Variance explained
print(f"PC1 variance: {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2 variance: {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total variance: {pca.explained_variance_ratio_.sum():.3f}")

# Component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
```

**Citation:** Pearson (1901), Jolliffe (2002)

---

## 3. Multivariate Analysis: Classification Models

### 3.1 Linear Discriminant Analysis (LDA)

**Purpose:** Linear classification with built-in dimensionality reduction

**Model Type:** Generative (models P(X|Y) and uses Bayes' rule)

**Assumptions:**

1. **Multivariate normality:** Features follow a multivariate Gaussian distribution within each class
2. **Homoscedasticity:** Equal covariance matrices across classes (Σ_human = Σ_LLM)
3. **Linear decision boundary:** Classes are separable by a hyperplane

**Status of Assumptions in Our Data:**

⚠️ **VIOLATED:**
- Univariate analysis showed non-normal distributions for most features
- Visual inspection (Q-Q plots, Shapiro-Wilk tests) confirmed significant deviations from normality
- Variance heterogeneity observed across classes

**Mathematical Foundation:**

LDA finds the projection direction w that maximizes the ratio of between-class to within-class scatter:

```
w* = argmax_w { (w^T S_B w) / (w^T S_W w) }
```

where:
- S_B = between-class scatter matrix = (μ_1 - μ_2)(μ_1 - μ_2)^T
- S_W = within-class scatter matrix = Σ_1 + Σ_2
- μ_1, μ_2 = class means
- Σ_1, Σ_2 = class covariance matrices

**Solution:**

```
w = S_W^(-1) (μ_1 - μ_2)
```

**Decision Rule:**

Classify new sample x as class 1 if:
```
w^T x > threshold
```

**Performance:**

ROC AUC = 94.12% (±0.17%) with 5-fold stratified cross-validation

**Why Lower Performance than Logistic Regression?**

The assumption violation (non-normality) reduces LDA's effectiveness. When data deviates from multivariate normality, LDA's generative approach (modeling P(X|Y)) suffers, while discriminative approaches (modeling P(Y|X) directly) remain robust.

**Implementation:**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X, y, cv=5, scoring='roc_auc')

print(f"LDA ROC AUC: {scores.mean():.4f} (±{scores.std():.4f})")
```

**Citation:** Fisher (1936), Hastie et al. (2009)

---

### 3.2 Logistic Regression

**Purpose:** Probabilistic binary classification with interpretable coefficients

**Model Type:** Discriminative (models P(Y|X) directly)

**Key Advantages:**

1. **No distributional assumptions on features:** Unlike LDA, Logistic Regression does not assume features are normally distributed
2. **Robust to non-normality:** Only assumes linear log-odds relationship
3. **Probabilistic outputs:** Provides calibrated class probabilities
4. **Interpretable coefficients:** Each β_i indicates the change in log-odds per unit increase in feature i

**Model Equation:**

```
log(P(Y=1|X) / P(Y=0|X)) = β₀ + β₁x₁ + β₂x₂ + ... + β₁₀x₁₀
```

where:
- Y=1 represents LLM class
- Y=0 represents human class
- β₀ = intercept
- β_i = coefficient for feature i

**Probability Prediction:**

```
P(Y=1|X) = 1 / (1 + exp(-(β₀ + Σ β_i x_i)))
```

This is the **logistic function** (sigmoid), ensuring probabilities lie in [0, 1].

**Decision Boundary:**

The decision boundary is linear in feature space (hyperplane where P(Y=1|X) = 0.5):

```
β₀ + β₁x₁ + ... + β₁₀x₁₀ = 0
```

**Regularization:**

No regularization used (C → ∞) because:
- Features were carefully selected (10 features, no redundancy)
- No evidence of overfitting in cross-validation
- Model complexity appropriate for sample size

**Training:**

Maximum likelihood estimation via iterative optimization (e.g., L-BFGS algorithm)

**Performance:**

ROC AUC = 97.03% (±0.14%) with 5-fold stratified cross-validation

**Why Best Performance?**

1. **Designed for classification:** Unlike PCA/LDA which emphasize dimensionality reduction
2. **Robust to non-normality:** Does not assume multivariate Gaussian features
3. **Flexible assumptions:** Only assumes linearity in log-odds space
4. **Discriminative approach:** Directly optimizes P(Y|X) rather than modeling P(X|Y)

**Implementation:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lr = LogisticRegression(max_iter=1000, random_state=42)
scores = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')

print(f"Logistic Regression ROC AUC: {scores.mean():.4f} (±{scores.std():.4f})")
```

**Coefficient Interpretation:**

After fitting, examine coefficients to understand feature importance:

```python
lr.fit(X_train, y_train)

# Coefficients
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr.coef_[0],
    'abs_coefficient': np.abs(lr.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(coef_df)
```

A positive coefficient β_i > 0 indicates that increasing feature i increases the log-odds of LLM class.

**Citation:** Hosmer & Lemeshow (2013), Bishop (2006)

---

### 3.3 Fuzzy Logic Classifier

**Documentation:** Complete details in `docs/fuzzy-classifier-documentation.md`

**Summary:**

A transparent, rule-based classifier using fuzzy membership functions to assign degrees of belonging to human or LLM categories.

**Key Design Principles:**

1. **Data-driven fuzzy sets:** Membership functions constructed from training data quantiles (33rd, 50th, 66th percentiles), not expert knowledge
2. **Triangular membership functions:** Simple, interpretable shapes with clear low/medium/high regions
3. **Automatic rule induction:** Rules derived from median comparisons (if median_human > median_LLM, higher values favor human)
4. **Takagi-Sugano order-zero inference:** Average aggregation of membership degrees

**Performance:**

ROC AUC = 89.34% (±0.04%) with 5-fold stratified cross-validation

**Key Characteristics:**

- **Interpretability:** Fully transparent decision-making via fuzzy rules
- **Robustness:** Lowest variance across folds (±0.04% vs ±0.14% for LogReg) indicates exceptional stability
- **Trade-off:** 7.7 percentage points AUC loss compared to Logistic Regression in exchange for full interpretability

**When to Use Fuzzy Classifier:**

- Regulatory contexts requiring transparent decisions
- Educational settings demonstrating classification logic
- Domains where understanding "why" matters more than marginal performance gains

**Implementation:**

```python
from fuzzy import FuzzyClassifier

# Train fuzzy classifier
fuzzy_clf = FuzzyClassifier(pos_label='human', neg_label='llm')
fuzzy_clf.fit(train_df, label_col='label')

# Predict probabilities
probs = fuzzy_clf.predict_proba(test_df)

# Predict classes
preds = fuzzy_clf.predict(test_df)
```

**Citation:** Zadeh (1965) - Fuzzy Sets, Takagi & Sugeno (1985) - Fuzzy Inference

---

## 4. Model Evaluation

### 4.1 Cross-Validation Strategy: Stratified K-Fold

**Method:** 5-fold stratified cross-validation

**Why Stratified?**

Stratification ensures each fold maintains the same class distribution as the original dataset:

1. **Balanced evaluation:** Prevents biased metrics from imbalanced folds
2. **Variance reduction:** More stable performance estimates across folds
3. **Standard practice:** Recommended for classification tasks

**Procedure:**

1. **Partition data:** Divide dataset into 5 folds while preserving 50/50 class balance in each fold
2. **Iterative training:** For each fold i = 1, ..., 5:
   - Train on folds {1, ..., 5} \ {i} (4 folds, 80% of data)
   - Test on fold i (1 fold, 20% of data)
3. **Aggregate metrics:** Compute mean and standard deviation of performance across 5 folds

**Why 5 Folds?**

- **Bias-variance trade-off:** 5-fold provides good balance
  - More folds (e.g., 10): Lower bias, higher variance
  - Fewer folds (e.g., 3): Higher bias, lower variance
- **Computational efficiency:** Reasonable training time
- **Standard in literature:** Widely adopted convention

**Random Seed:** 42 (for reproducibility)

**Implementation:**

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"Mean: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
```

**Citation:** Kohavi (1995), Hastie et al. (2009)

---

### 4.2 Performance Metric: ROC AUC

**Metric:** Area Under the Receiver Operating Characteristic Curve (ROC AUC)

**Why ROC AUC?**

1. **Threshold-independent:** Evaluates classifier across all possible classification thresholds, not just a single cutoff
2. **Balanced dataset appropriate:** Performs well on balanced data (our dataset: 50% human, 50% LLM)
3. **Probabilistic interpretation:** AUC represents the probability that the classifier ranks a random positive instance higher than a random negative instance
4. **Robust to slight imbalance:** Less sensitive than accuracy to minor class imbalance
5. **Standard in binary classification:** Widely adopted metric in machine learning research

**ROC Curve:**

The ROC curve plots True Positive Rate (TPR) vs False Positive Rate (FPR) at various thresholds:

- **True Positive Rate (Sensitivity, Recall):**
  ```
  TPR = TP / (TP + FN)
  ```

- **False Positive Rate:**
  ```
  FPR = FP / (FP + TN)
  ```

**AUC Interpretation:**

- **AUC = 1.0:** Perfect classifier (100% TPR, 0% FPR)
- **AUC = 0.9-1.0:** Excellent discrimination
- **AUC = 0.8-0.9:** Good discrimination
- **AUC = 0.7-0.8:** Acceptable discrimination
- **AUC = 0.5:** Random classifier (no discrimination)
- **AUC < 0.5:** Worse than random (inverted predictions)

**Probabilistic Meaning:**

AUC = P(score(positive) > score(negative))

For our study: AUC = 97% means that 97% of the time, a randomly chosen LLM text receives a higher LLM-probability score than a randomly chosen human text.

**Reporting Format:**

Mean ± Standard Deviation across 5 cross-validation folds:

Example: ROC AUC = 0.9703 ± 0.0014 (97.03% ± 0.14%)

**Alternative Metrics Considered (and Why Not Used):**

| Metric | Why Not Primary Metric |
|--------|------------------------|
| **Accuracy** | Too simplistic; hides threshold choice; inflated by balanced data |
| **F1-score** | Requires threshold selection; sensitive to class imbalance |
| **Precision/Recall** | Threshold-dependent; incomplete picture alone |
| **PR AUC** | Better for imbalanced data, but our data is balanced |

**Implementation:**

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)
# Or directly:
roc_auc = roc_auc_score(y_test, y_proba)

print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**Citation:** Provost & Fawcett (2001), Fawcett (2006)

---

## 5. Results Summary

### 5.1 Univariate Statistical Tests

**Sample Size:** n = 20,000 (subsample from EDA.ipynb for statistical tests)

**Results Table:**

| Feature | Median (Human) | Median (LLM) | Cliff's δ | Effect Size | p-value (adj) | Significant |
|---------|----------------|--------------|-----------|-------------|---------------|-------------|
| **char_entropy** | 4.560 | 4.254 | **-0.881** | **Large** | < 0.001 | ✓ |
| **sent_std** | 12.487 | 4.528 | **-0.790** | **Large** | < 0.001 | ✓ |
| **sent_burst** | 0.640 | 0.319 | **-0.663** | **Large** | < 0.001 | ✓ |
| **ttr** | 0.570 | 0.735 | +0.616 | **Large** | < 0.001 | ✓ |
| **hapax_prop** | 0.417 | 0.581 | +0.564 | **Large** | < 0.001 | ✓ |
| **herdan_c** | 0.903 | 0.929 | +0.450 | Medium | < 0.001 | ✓ |
| **bigram_repeat_ratio** | 0.066 | 0.030 | -0.424 | Medium | < 0.001 | ✓ |
| **func_word_ratio** | 0.313 | 0.347 | +0.378 | Medium | < 0.001 | ✓ |
| **sent_mean** | 20.0 | 16.5 | -0.290 | Small | < 0.001 | ✓ |
| **first_person_ratio** | 0.0025 | 0.0 | -0.049 | Negligible | < 0.001 | ✓ |
| **fk_grade** | 0.0 | 0.0 | 0.0 | Negligible | 1.000 | ✗ |

*Note: All p-values adjusted using Benjamini-Hochberg FDR correction*

**Key Findings:**

1. **10 out of 11 features** show statistically significant differences (p < 0.05 after FDR correction)
2. **6 features** exhibit **large effect sizes** (|δ| ≥ 0.474):
   - Character entropy (δ = -0.881)
   - Sentence standard deviation (δ = -0.790)
   - Burstiness (δ = -0.663)
   - Type-token ratio (δ = +0.616)
   - Hapax proportion (δ = +0.564)
3. **Strongest discriminators:** Character entropy and sentence variability metrics
4. **Flesch-Kincaid grade:** Not applicable for Portuguese texts (set to 0)

**Interpretation:**

Negative δ values indicate human texts have higher median values (e.g., character entropy, sentence variability). Positive δ values indicate LLM texts have higher median values (e.g., lexical diversity metrics like TTR).

---

### 5.2 Multivariate Classification Models

**Dataset:** Balanced dataset, n = 100,000 (50,000 human, 50,000 LLM)

**Cross-Validation:** 5-fold stratified

**Metric:** ROC AUC (mean ± standard deviation)

**Model Comparison:**

| Model | ROC AUC | Std Dev | Pros | Cons |
|-------|---------|---------|------|------|
| **Logistic Regression** | **97.03%** | ±0.14% | Best performance, robust to non-normality | Black-box, less interpretable |
| **LDA** | 94.12% | ±0.17% | Fast, built-in dimensionality reduction | Assumption violation (non-normality) |
| **Fuzzy Classifier** | 89.34% | ±0.04% | **Fully interpretable**, lowest variance | 7.7% AUC loss vs LogReg |

**Performance Analysis:**

1. **Logistic Regression (winner):**
   - Highest AUC: 97.03%
   - Robust to non-normal feature distributions
   - Discriminative approach (models P(Y|X) directly)

2. **LDA (runner-up):**
   - Good AUC: 94.12% (2.9 pp below LogReg)
   - Suffers from assumption violation
   - Generative approach sensitive to distributional mismatch

3. **Fuzzy Classifier (interpretable alternative):**
   - Acceptable AUC: 89.34% (7.7 pp below LogReg)
   - **Exceptional stability:** σ = ±0.04% (3.5× lower than LogReg)
   - Full transparency: decision process completely interpretable

**Trade-off Analysis:**

- **Logistic Regression vs Fuzzy:** Gain 7.7% AUC by sacrificing interpretability
- **Fuzzy advantage:** Lowest variance (most stable predictions across folds)
- **Use case dependent:** Choose Fuzzy for transparency, LogReg for performance

---

## 6. Reconciliation: Non-Parametric Tests vs Parametric Models

**Apparent Contradiction:**

Why use **non-parametric tests** (Mann-Whitney U) for univariate analysis but **parametric models** (Logistic Regression, LDA) for multivariate classification?

**Answer: Different Purposes, Different Assumptions**

### Univariate Analysis (Mann-Whitney U Test)

**Purpose:** Determine if individual features are discriminant

**Assumptions:**
- Independent observations
- Ordinal or continuous data
- **No assumption about distribution shape**

**Why non-parametric:**
- Features exhibit non-normal distributions (skewness, heavy tails, outliers)
- Mann-Whitney U test has **higher statistical power** than t-test when normality is violated
- Provides robust hypothesis testing without distribution assumptions

**Goal:** Hypothesis testing (does an effect exist?)

### Multivariate Analysis (Logistic Regression, LDA)

**Purpose:** Build predictive models combining all features

**Different assumptions:**

**LDA Assumptions:**
- Multivariate Gaussian distribution within each class
- Equal covariance matrices
- **STATUS:** Violated in our data → explains lower performance (94.12%)

**Logistic Regression Assumptions:**
- **Only assumes linear log-odds:** P(Y|X) = logistic(β₀ + Σ β_i x_i)
- **No assumption about feature distributions**
- **Robust to non-normality** because it models P(Y|X) directly (discriminative)

**Goal:** Prediction/classification (maximize AUC)

### Key Insight: Why Logistic Regression Works Despite Non-Normality

Logistic Regression is a **discriminative model** that optimizes P(Y|X) directly:

1. **No feature distribution assumptions:** Only models the decision boundary
2. **Flexible:** Can handle non-normal, skewed, or multimodal features
3. **Robust:** Outliers affect the decision boundary but don't invalidate the model

In contrast:
- **LDA** is a **generative model** that assumes P(X|Y) ~ N(μ, Σ)
- When this assumption fails, LDA's performance degrades
- Our results confirm this: LDA (94.12%) < LogReg (97.03%)

### Empirical Validation

The superior performance of Logistic Regression (97.03%) vs LDA (94.12%) **empirically validates** our methodological choice:

- Non-parametric tests for univariate analysis (correct choice given non-normality)
- Discriminative model (LogReg) for multivariate classification (robust to non-normality)
- Generative model (LDA) suffers from assumption violation (as expected)

**Conclusion:**

There is **no contradiction**. The choice of statistical method depends on:
1. **Purpose:** Hypothesis testing vs prediction
2. **Assumptions:** Distribution-free vs specific distributional assumptions
3. **Robustness:** Sensitivity to violations

Our methodology appropriately matches method to purpose and data characteristics.

---

## 7. References

### Statistical Tests

- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50-60.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494-509.
- Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys? *Annual Meeting of the Florida Association of Institutional Research*, 1-33.

### Multiple Comparisons

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

### Dimensionality Reduction

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559-572.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.

### Classification Models

- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

### Cross-Validation

- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *International Joint Conference on Artificial Intelligence (IJCAI)*, 14(2), 1137-1145.

### Evaluation Metrics

- Provost, F., & Fawcett, T. (2001). Robust classification for imprecise environments. *Machine Learning*, 42(3), 203-231.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.

### Fuzzy Logic

- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.
- Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. *IEEE Transactions on Systems, Man, and Cybernetics*, 15(1), 116-132.

---

## 8. Implementation Details

### File Locations

- **Statistical tests:** `src/tests.py`
- **Multivariate models:** `src/models.py`
- **Fuzzy classifier:** `src/fuzzy.py`
- **Exploratory analysis:** `EDA.ipynb`

### Command-Line Usage

**Run statistical tests:**
```bash
python src/tests.py --features features.csv --label-col label --out statistical_tests_results.csv
```

**Run PCA:**
```bash
python src/models.py pca --features features.csv --label-col label --out pca_scores.csv --plot pca_scatter.png
```

**Evaluate classifiers:**
```bash
python src/models.py classify --features features.csv --label-col label --n-splits 5 --roc-out roc_results.pkl --pr-out pr_results.pkl
```

### Results Files

- `statistical_tests_results.csv` - Mann-Whitney U test results with Cliff's delta and FDR-adjusted p-values
- `eda_results_for_paper.json` - Comprehensive EDA results for paper
- `roc_results.pkl` - ROC curves and AUC values for LDA and Logistic Regression
- `pr_results.pkl` - Precision-Recall curves for LDA and Logistic Regression
- `fuzzy_roc_results.pkl` - ROC curves for Fuzzy Classifier

---

**Last Updated:** 2025-12-06

**Document Purpose:** Complete transparency and reproducibility of all statistical methods used in the study.
