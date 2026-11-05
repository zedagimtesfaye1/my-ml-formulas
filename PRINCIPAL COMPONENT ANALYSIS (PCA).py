# ============================================================================
# FORMULA 08: PCA  (PRINCIPAL COMPONENT ANALYSIS) — UNSUPERVISED
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Wuchale
# Task   : Compress many features into a few — no labels
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "PCA looks complicated. It is not. It is FOUR lines.
#  Line 1: SUBTRACT the average — center the data.
#  Line 2: Find the directions where data is spread out most.
#  Line 3: Those directions are the principal components.
#  Line 4: Multiply. Z = Xc · W. Done.
#
#  For Wuchale: 30,000 students, each with 10 features.
#  Line 4 outputs 2 numbers per student. Two. Down from ten.
#  And 90% of the truth survives.
#
#  That is PCA. Four lines. One chart. The whole story on one page."
#
# ============================================================================
# THE COMPLETE EXACT FORMULA — 4 LINES
# ============================================================================
#
#   LINE 1 — Center the data:
#       Xc = X − mean(X)
#
#   LINE 2 — Covariance matrix:
#       C = (1/(n−1)) · Xcᵀ · Xc
#
#   LINE 3 — Eigen decomposition (find principal directions):
#       C · v = λ · v
#       → eigenvectors v₁, v₂, ..., vₚ (sorted by λ, largest first)
#       → W = [v₁, v₂, ..., vₖ]   (top k eigenvectors as columns)
#
#   LINE 4 — Project (the answer):
#       Z = Xc · W
#
#   VARIANCE EXPLAINED by each PC:
#       variance_k = λ_k / Σ(λ)
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   X     = original data (n samples × p features)
#   Xc    = centered data (X minus column means)
#   C     = covariance matrix (p × p)
#   λ     = eigenvalue (variance along that direction)
#   v     = eigenvector (direction)
#   W     = matrix of top-k eigenvectors (p × k)
#   Z     = projected/compressed data (n × k)   ← YOUR "Z = X·V"
#   n     = number of samples
#   p     = number of original features
#   k     = number of components kept (k < p)
#   Σ(λ)  = sum of all eigenvalues
#   ᵀ     = transpose
#
# ============================================================================
# WHY IT MATTERS
# ============================================================================
#
#   PCA replaces 10 features with 2 "super-features" that keep 90%
#   of the information. You go from "can't see" to "can see".
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT WUCHALE DATA (5 students, 2 features)
# ============================================================================
#
#   DATA:
#     Student  Distance(X₁)  Age(X₂)
#        A          2          18
#        B          4          20
#        C          6          22
#        D          8          24
#        E         10          24
#
#   LINE 1 — CENTER:
#     mean(X₁) = (2+4+6+8+10)/5 = 6
#     mean(X₂) = (18+20+22+24+24)/5 = 21.6
#
#     Xc (centered):
#       A: (-4.0, -3.6)
#       B: (-2.0, -1.6)
#       C: ( 0.0,  0.4)
#       D: ( 2.0,  2.4)
#       E: ( 4.0,  2.4)
#
#   LINE 2 — COVARIANCE MATRIX C:
#     C[0][0] = Σ(X₁c²)/(n−1)         = (16+4+0+4+16)/4 = 10.0
#     C[0][1] = Σ(X₁c·X₂c)/(n−1)      = (14.4+3.2+0+4.8+9.6)/4 = 8.0
#     C[1][1] = Σ(X₂c²)/(n−1)         = (12.96+2.56+0.16+5.76+5.76)/4 = 6.8
#
#     C = [[10.0,  8.0],
#          [ 8.0,  6.8]]
#
#   LINE 3 — EIGEN:
#     Trace = 16.8,  Det = 4
#     λ₁ = 16.559   (variance along PC1)
#     λ₂ =  0.241   (variance along PC2)
#     v₁ = (0.773, 0.634)
#
#     W = [[0.773],
#          [0.634]]              ← keeping only PC1
#
#   LINE 4 — PROJECT  Z = Xc · W:
#     A: -5.37
#     B: -2.56
#     C: +0.25
#     D: +3.07
#     E: +4.61
#
#   VARIANCE KEPT by PC1:
#     16.559 / (16.559 + 0.241) = 98.6%
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the data matrix. NEVER edit math functions.
#   Rows = samples. Columns = features. All rows same length.
# ============================================================================


import numpy as np

print("=" * 68)
print("FORMULA 08: PCA")
print("=" * 68)


# ============================================================================
# PART 1: MANUAL VERSION (2 features — transparent math)
# ============================================================================

# ---------- YOUR DATA (ONLY EDIT THIS) ----------
data = np.array([
    [ 2, 18],
    [ 4, 20],
    [ 6, 22],
    [ 8, 24],
    [10, 24],
], dtype=float)

# ---------- LINE 1: CENTER (NEVER EDIT) ----------
mean = data.mean(axis=0)
Xc   = data - mean

print("\n--- MANUAL VERSION (5 students, 2 features) ---")
print(f"\nLINE 1 — Center. Means = (distance {mean[0]:.1f}, age {mean[1]:.1f})")
print("Xc:")
print(Xc)

# ---------- LINE 2: COVARIANCE (NEVER EDIT) ----------
n = data.shape[0]
C = (Xc.T @ Xc) / (n - 1)

print("\nLINE 2 — Covariance matrix C:")
print(C)

# ---------- LINE 3: EIGEN (NEVER EDIT) ----------
eigenvalues, eigenvectors = np.linalg.eigh(C)
order = np.argsort(eigenvalues)[::-1]         # sort descending
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\nLINE 3 — Eigenvalues (sorted descending):")
print(eigenvalues)
print("Eigenvectors (columns):")
print(eigenvectors)

# Variance explained
total = eigenvalues.sum()
print("\nVariance explained:")
for k in range(len(eigenvalues)):
    print(f"  PC{k+1}: {eigenvalues[k]/total*100:.2f}%")

# ---------- LINE 4: PROJECT (YOUR Z = X·V) ----------

# Keep only PC1
W = eigenvectors[:, :1]                       # p × 1
Z = Xc @ W                                    # THIS IS YOUR Z = Xc · W

print("\nLINE 4 — W (top 1 eigenvector):")
print(W)

print("\nLINE 4 — Projected Z = Xc · W  (your formula!):")
for i in range(n):
    print(f"  Student {chr(65+i)}: Z = {Z[i,0]:+.4f}")


# ============================================================================
# PART 2: LIBRARY VERSION (scikit-learn on 4 features)
# ============================================================================
#
#   Standardize first when features have different scales.
#   (age ~20, income ~7000 → income would dominate without scaling)
#
# ============================================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 68)
print("LIBRARY VERSION (scikit-learn PCA on 4-feature Wuchale data)")
print("=" * 68)

# ---------- YOUR DATA (ONLY EDIT THIS) ----------
# Columns = [distance, age, income_k, score]
wuchale = np.array([
    [ 2, 16, 5, 75],
    [ 4, 17, 4, 70],
    [ 6, 18, 6, 80],
    [ 8, 20, 5, 72],
    [10, 22, 7, 85],
    [15, 25, 8, 78],
    [18, 28, 9, 82],
    [20, 30, 7, 76],
], dtype=float)

feature_names = ["distance", "age", "income", "score"]

# ---------- STANDARDIZE (NEVER EDIT) ----------
scaler = StandardScaler()
X_std  = scaler.fit_transform(wuchale)

# ---------- PCA (NEVER EDIT) ----------
pca = PCA(n_components=2)
Z   = pca.fit_transform(X_std)                # SAME AS Z = Xc·W internally

print("\nVariance kept by each PC:")
for k, r in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{k+1}: {r*100:.2f}%")
print(f"  TOTAL: {pca.explained_variance_ratio_.sum()*100:.2f}%")

print("\nPC1 weights (how each feature contributes):")
for name, w in zip(feature_names, pca.components_[0]):
    print(f"  {name:<10} → {w:+.3f}")

print("\nPC2 weights:")
for name, w in zip(feature_names, pca.components_[1]):
    print(f"  {name:<10} → {w:+.3f}")

print("\nCompressed Z (PC1, PC2) per student:")
for i in range(len(wuchale)):
    print(f"  S{i+1}: ({Z[i,0]:+.3f}, {Z[i,1]:+.3f})")


# ============================================================================
# PART 3: SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Manual (2 features) — PC1 keeps {eigenvalues[0]/total*100:.1f}% of variation")
print(f"  Library (4 features) — PC1+PC2 keep "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}% of variation")
print()
print("  YOUR FORMULA:")
print("    Z = Xc · W")
print("    (Xc = X centered, W = top eigenvectors)")
print("    → 4 features shrink to 2 without losing the story.")
print()
print("  DECISION FOR WUCHALE:")
print("  → Plot all 30,000 students on one 2D chart.")
print("  → PC1 (ruralness) separates reachable vs remote.")
print("  → PC2 (academic strength) sorts strong candidates.")
print("  → TWO strategies become visually obvious from ONE chart.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - Xc = data - mean
#      - C = (Xc.T @ Xc) / (n-1)
#      - np.linalg.eigh(C)
#      - StandardScaler().fit_transform(...)
#      - PCA(n_components=K).fit_transform(...)
#
#   🔁 ALWAYS edit:
#      - data matrix (rows = samples, columns = features)
#      - n_components (how many PCs to keep)
#
#   ⚙️ AUTO-computed:
#      - mean, Xc, C, λ, eigenvectors, W, Z
#      - variance_explained_ratio
#
#   ⚠️ If features have different scales, ALWAYS standardize first.
#
# ============================================================================
# WUCHALE SOLVED BY:  K-Means (F07) for grouping  +  PCA (F08) for visualization
# ============================================================================
