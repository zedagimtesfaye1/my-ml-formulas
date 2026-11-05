# ============================================================================
# FORMULA 17: EIGENVALUES AND EIGENVECTORS
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga + Wuchale
# Task   : Find the "natural directions" of a matrix — the heart of PCA
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Most matrices are boring. You multiply a matrix by a vector, and the
#  vector TURNS. It changes direction. It rotates.
#
#  But a few special vectors do NOT turn. You multiply them by the matrix
#  and they just get LONGER or SHORTER. Same direction. Only size changes.
#
#  Those special vectors are called EIGENVECTORS. The amount they get
#  longer or shorter is the EIGENVALUE. The word comes from German:
#  EIGEN means OWN. These are the matrix's OWN vectors — the directions
#  it truly cares about.
#
#  Why does this matter? Because eigenvalues reveal the SECRET STRUCTURE
#  of any dataset.
#
#  Think about Jigjiga students. We have two features: distance and age.
#  Are they independent? No — older students live farther. There is a
#  HIDDEN direction where the data is stretched. PCA finds that direction
#  by finding the EIGENVECTORS of the covariance matrix. The eigenvalue
#  tells you HOW STRETCHED that direction is. Large eigenvalue = the
#  data varies a lot along that direction. That is where the information
#  lives.
#
#  Everywhere in machine learning:
#    - PCA uses eigenvalues to rank components by importance.
#    - Recommendation systems use them to find hidden user interests.
#    - Google's PageRank uses the principal eigenvector to rank pages.
#    - Vibration analysis, quantum mechanics, finance — everywhere.
#    - When you compress a face, you find its eigenfaces.
#    - When you compress a song, you find its eigenfrequencies.
#
#  For Wuchale: 30,000 students. Eigenvectors of the covariance matrix
#  tell us the TRUE axes — not the raw features. The first eigenvector
#  might be 'ruralness + affordability' — a combination that reveals
#  the pattern more clearly than distance or income alone.
#
#  And here is the beautiful thing. Eigenvalues are RANKED. The biggest
#  one carries the most information. The second carries most of what
#  is left. The smallest carry noise. You can KEEP the first few and
#  THROW AWAY the rest — losing almost nothing.
#
#  That is why PCA works. That is why eigenvalues matter. They tell
#  you WHERE to look and HOW MUCH to keep. Without them, you are
#  drowning in features. With them, you see the truth."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA — EIGENVALUE EQUATION
# ============================================================================
#
#   THE EIGENVALUE EQUATION:
#
#       A · v = λ · v
#
#   where:
#       A = square matrix (n × n)
#       v = eigenvector (the special direction)      ← we want this
#       λ = eigenvalue (scalar, stretch factor)      ← we want this
#
#   Find eigenvalues by solving the CHARACTERISTIC EQUATION:
#
#       det( A − λ·I ) = 0
#
#   where I = identity matrix (same size as A)
#
#   Once you have λ, find v by solving:
#
#       ( A − λ·I ) · v = 0
#
#   Then normalize to unit length:
#
#       v_normalized = v / ||v||
#
#   where ||v|| = √(v₁² + v₂² + ... + vₙ²)
#
# ============================================================================
# THE 2×2 CLOSED FORM (memorize this — no software needed)
# ============================================================================
#
#   For matrix:
#       A = [[ a,  b ],
#            [ b,  d ]]      (symmetric 2×2 — the most common case)
#
#   EIGENVALUES (solve λ² − (a+d)·λ + (ad−b²) = 0):
#
#       λ = [ (a+d)  ±  √( (a+d)² − 4·(ad−b²) ) ] / 2
#
#   In simpler terms:
#       Trace = a + d
#       Det   = a·d − b²
#       λ₁ = (Trace + √(Trace² − 4·Det)) / 2
#       λ₂ = (Trace − √(Trace² − 4·Det)) / 2
#
#   EIGENVECTOR for λ:
#       If b ≠ 0:   v = (b,  λ − a)     (then normalize)
#       If b = 0:   eigenvalues are a, d; eigenvectors are (1,0) and (0,1)
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   A           = square matrix (covariance matrix in PCA)
#   v           = eigenvector (unit-length direction)
#   λ           = eigenvalue (stretch factor along v)
#   I           = identity matrix
#   det         = determinant
#   Trace       = sum of diagonal entries (a + d for 2×2)
#   ||v||       = length of vector v
#   ±           = plus or minus (two solutions → two eigenvalues)
#   a,b,d       = entries of symmetric 2×2 matrix
#
# ============================================================================
# WHY IT MATTERS — THE SECRET BEHIND PCA
# ============================================================================
#
#   In PCA (Formula 08), we computed:
#       C = covariance matrix (p × p)
#       eigenvalues, eigenvectors = np.linalg.eigh(C)
#
#   Those eigenvalues + eigenvectors ARE this formula.
#   PCA = this formula + one projection step.
#
#   For Wuchale, this is what happens:
#       1. Compute covariance matrix of students (distance, age)
#       2. Find its eigenvalues + eigenvectors   ← THIS FORMULA
#       3. Big eigenvalue = the main direction of variation
#       4. Project students onto that direction  ← Z = Xc · v
#
#   Without eigenvalues, no PCA. Without PCA, no dimensionality
#   reduction. Without dimensionality reduction, no modern AI.
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA DATA
# ============================================================================
#
#   We use the covariance matrix from Formula 08 (PCA):
#       C = [[10.0,  8.0],
#            [ 8.0,  6.8]]
#
#   STEP 1 — Compute Trace and Det:
#       Trace = 10.0 + 6.8 = 16.8
#       Det   = 10.0·6.8 − 8.0·8.0 = 68 − 64 = 4.0
#
#   STEP 2 — Solve characteristic equation  λ² − Trace·λ + Det = 0:
#       λ² − 16.8·λ + 4.0 = 0
#
#       λ = (16.8 ± √(16.8² − 4·4.0)) / 2
#         = (16.8 ± √(282.24 − 16)) / 2
#         = (16.8 ± √266.24) / 2
#         = (16.8 ± 16.317) / 2
#
#       λ₁ = (16.8 + 16.317) / 2 = 16.559   ← LARGE (main direction)
#       λ₂ = (16.8 − 16.317) / 2 =  0.241   ← small (noise)
#
#   STEP 3 — Find eigenvector for λ₁ = 16.559:
#       (C − λ₁·I) · v = 0
#
#       [[10.0 − 16.559,   8.0     ],   [v₁]   [0]
#        [  8.0,          6.8 − 16.559]] · [v₂] = [0]
#
#       [[-6.559,  8.0    ],   [v₁]   [0]
#        [ 8.0,   -9.759  ]] · [v₂] = [0]
#
#       From first row:  -6.559·v₁ + 8.0·v₂ = 0
#                        v₂ = (6.559 / 8.0)·v₁ = 0.820·v₁
#
#       Let v₁ = 1, then v₂ = 0.820
#       Raw vector: v = (1, 0.820)
#
#       Normalize:  ||v|| = √(1² + 0.820²) = √(1 + 0.672) = √1.672 = 1.293
#       v₁ = (1 / 1.293, 0.820 / 1.293) = (0.773, 0.634)
#
#   STEP 4 — Find eigenvector for λ₂ = 0.241:
#       From second row of (C − λ₂·I):  (6.8 − 0.241)·v₂ + 8.0·v₁ = 0
#         6.559·v₂ + 8.0·v₁ = 0
#         v₂ = −(8.0 / 6.559)·v₁ = −1.220·v₁
#
#       Let v₁ = 1, then v₂ = −1.220
#       Raw: v = (1, −1.220)
#       ||v|| = √(1 + 1.489) = √2.489 = 1.578
#       v₂ = (0.634, −0.773)
#
#   VERIFICATION:
#       C · v₁ = [[10, 8], [8, 6.8]] · [0.773, 0.634]ᵀ
#              = [10·0.773 + 8·0.634,  8·0.773 + 6.8·0.634]
#              = [7.73 + 5.072,  6.184 + 4.311]
#              = [12.802, 10.495]
#
#       λ₁ · v₁ = 16.559 · [0.773, 0.634] = [12.802, 10.495]  ✅ MATCH!
#
#   INTERPRETATION:
#       λ₁ = 16.559   → 82.6% of the variation
#       λ₂ = 0.241    →  1.4% of the variation
#       v₁ = (0.773, 0.634)  →  "distance + age together" (main direction)
#       v₂ = (0.634, −0.773) →  "distance vs age" (orthogonal direction)
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the matrix A (must be square).
#   NEVER edit the eigenvalue solver.
#   For symmetric matrices (covariance), use numpy.linalg.eigh (more stable).
# ============================================================================


import numpy as np
import math


# ============================================================================
# PART 1: MANUAL EIGENVALUES/EIGENVECTORS — 2×2 closed form
# ============================================================================

print("=" * 68)
print("PART 1: EIGENVALUES/EIGENVECTORS (manual, 2×2)")
print("=" * 68)

# ---------- MATRIX (ONLY EDIT THIS) ----------
# Covariance matrix from Formula 08 (Wuchale)
A = np.array([[10.0,  8.0],
              [ 8.0,  6.8]])

print("\nMatrix A:")
print(A)

# ---------- STEP 1: TRACE AND DET ----------

a = A[0, 0]
b = A[0, 1]
d = A[1, 1]

trace = a + d
det   = a * d - b * b

print(f"\nSTEP 1: Trace = {trace}   Det = {det}")

# ---------- STEP 2: EIGENVALUES via quadratic formula ----------

discriminant = trace * trace - 4 * det
sqrt_disc = math.sqrt(discriminant)

lam1 = (trace + sqrt_disc) / 2
lam2 = (trace - sqrt_disc) / 2

print(f"\nSTEP 2: Characteristic equation: λ² − {trace}·λ + {det} = 0")
print(f"        Discriminant = {trace}² − 4·{det} = {discriminant}")
print(f"        λ₁ = ({trace} + {sqrt_disc:.4f}) / 2 = {lam1:.4f}")
print(f"        λ₂ = ({trace} − {sqrt_disc:.4f}) / 2 = {lam2:.4f}")

# ---------- STEP 3: EIGENVECTORS (unit length) ----------

def eigenvector_2x2(lam, a, b, d):
    # Solve (A − λ·I)·v = 0
    if b != 0:
        # From first row:  (a − λ)·v1 + b·v2 = 0
        #                  v2 = −(a − λ)/b · v1 = (λ − a)/b · v1
        v1_raw = 1.0
        v2_raw = (lam - a) / b
    else:
        # Diagonal matrix — eigenvectors are standard basis
        if abs(lam - a) < abs(lam - d):
            v1_raw, v2_raw = 1.0, 0.0
        else:
            v1_raw, v2_raw = 0.0, 1.0

    norm = math.sqrt(v1_raw ** 2 + v2_raw ** 2)
    return np.array([v1_raw / norm, v2_raw / norm])


v1 = eigenvector_2x2(lam1, a, b, d)
v2 = eigenvector_2x2(lam2, a, b, d)

print(f"\nSTEP 3: Eigenvector for λ₁ = {lam1:.4f}:")
print(f"        v₁ = {v1}")

print(f"\nSTEP 3: Eigenvector for λ₂ = {lam2:.4f}:")
print(f"        v₂ = {v2}")

# ---------- STEP 4: VERIFICATION  A·v = λ·v ----------

print("\nSTEP 4: Verification  A·v = λ·v")

Av1 = A @ v1
lv1 = lam1 * v1
print(f"  A · v₁     = {Av1}")
print(f"  λ₁ · v₁    = {lv1}")
print(f"  Match:     {np.allclose(Av1, lv1)}")

Av2 = A @ v2
lv2 = lam2 * v2
print(f"\n  A · v₂     = {Av2}")
print(f"  λ₂ · v₂    = {lv2}")
print(f"  Match:     {np.allclose(Av2, lv2)}")

# ---------- STEP 5: VARIANCE EXPLAINED ----------

total_var = lam1 + lam2
print(f"\nSTEP 5: Variance explained")
print(f"  λ₁ / (λ₁+λ₂) = {lam1/total_var*100:.2f}%   (main direction)")
print(f"  λ₂ / (λ₁+λ₂) = {lam2/total_var*100:.2f}%   (residual)")


# ============================================================================
# PART 2: LIBRARY EIGEN — numpy.linalg.eigh (symmetric matrices)
# ============================================================================
#
#   np.linalg.eigh is for SYMMETRIC matrices (like covariance).
#     → returns eigenvalues in ASCENDING order
#     → eigenvalues are GUARANTEED REAL (not complex)
#     → faster and more stable than general eig
#
#   np.linalg.eig is the general solver.
#     → handles any matrix
#     → may return COMPLEX eigenvalues for non-symmetric matrices
#     → for covariance matrices, always use eigh
#
# ============================================================================

print("\n" + "=" * 68)
print("PART 2: EIGEN — numpy.linalg.eigh")
print("=" * 68)

eigenvalues, eigenvectors = np.linalg.eigh(A)

# eigh returns ASCENDING — reverse for DESCENDING
order = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\nEigenvalues (descending):")
for i, lam in enumerate(eigenvalues):
    print(f"  λ{i+1} = {lam:.4f}")

print("\nEigenvectors (columns, one per eigenvalue):")
print(eigenvectors)

print("\nVerification A·v = λ·v (all together):")
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  λ{i+1}: A·v = {A @ v}, λ·v = {lam * v}, match = {np.allclose(A @ v, lam * v)}")


# ============================================================================
# PART 3: WUCHALE 4-FEATURE EXAMPLE — where eigenvectors reveal the truth
# ============================================================================
#
#   For a bigger case, we build a covariance matrix from real data
#   and find its eigenvectors. The matrix must be SQUARE.
#
# ============================================================================

print("\n" + "=" * 68)
print("PART 3: WUCHALE 4-FEATURE COVARIANCE + EIGEN")
print("=" * 68)

# Wuchale students: [distance, age, income_k, score]
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

# Standardize (eigenvalues of unnormalized covariance are misleading)
mean = wuchale.mean(axis=0)
std  = wuchale.std(axis=0)
X_std = (wuchale - mean) / std

# Covariance matrix (p × p) — symmetric
p = X_std.shape[1]
Cov = (X_std.T @ X_std) / (len(X_std) - 1)

print("\nCovariance matrix (standardized):")
np.set_printoptions(precision=3, suppress=True)
print(Cov)

# Eigen decomposition
eigenvalues4, eigenvectors4 = np.linalg.eigh(Cov)

# Sort descending
order4 = np.argsort(eigenvalues4)[::-1]
eigenvalues4  = eigenvalues4[order4]
eigenvectors4 = eigenvectors4[:, order4]

print("\nEigenvalues (descending):")
for i, lam in enumerate(eigenvalues4):
    pct = lam / eigenvalues4.sum() * 100
    print(f"  λ{i+1} = {lam:.4f}  ({pct:.2f}% of variation)")

feature_names = ["distance", "age", "income", "score"]

print("\nEigenvector 1 (the main direction):")
for name, w in zip(feature_names, eigenvectors4[:, 0]):
    print(f"  {name:<10} → {w:+.4f}")

print("\nEigenvector 2:")
for name, w in zip(feature_names, eigenvectors4[:, 1]):
    print(f"  {name:<10} → {w:+.4f}")


# ============================================================================
# PART 4: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print("  THE EIGENVALUE EQUATION:  A · v = λ · v")
print()
print("  JIGJIGA (Wuchale) 2×2 case:")
print(f"    λ₁ = {lam1:.4f}   (82.6% of variation)")
print(f"    λ₂ = {lam2:.4f}   (1.4% of variation)")
print(f"    v₁ = ({v1[0]:.3f}, {v1[1]:.3f})   ← main direction")
print(f"    v₂ = ({v2[0]:.3f}, {v2[1]:.3f})   ← orthogonal direction")
print()
print("  MEANING:")
print("  → v₁ is the direction where data varies MOST.")
print("  → λ₁ is HOW MUCH it varies there.")
print("  → Together they define the 'natural axis' of the data.")
print("  → For PCA: keep the axis with biggest λ, throw the rest.")
print()
print("  DECISION FOR JIGJIGA / WUCHALE:")
print(f"  → The first eigenvector explains "
      f"{eigenvalues4[0]/eigenvalues4.sum()*100:.1f}% of all variation.")
print(f"  → The second adds {eigenvalues4[1]/eigenvalues4.sum()*100:.1f}%.")
print(f"  → Together they capture the essential pattern of 4 features.")
print(f"  → We can plot 30,000 students on a 2D chart using ONLY v₁ and v₂.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - eigenvector_2x2 function
#      - np.linalg.eigh / np.linalg.eig calls
#      - the sorting code (argsort descending)
#
#   🔁 ALWAYS edit:
#      - matrix A (must be square)
#      - wuchale data (rows = samples, columns = features)
#
#   ⚙️ AUTO-computed:
#      - trace, det
#      - λ₁, λ₂, v₁, v₂
#      - variance explained per eigenvector
#      - verification A·v = λ·v
#
#   ⚠️ Use eigh for SYMMETRIC matrices (covariance, correlation)
#   ⚠️ Use eig for GENERAL matrices (may return complex eigenvalues)
#   ⚠️ eigh returns ASCENDING — reverse for descending
#   ⚠️ Eigenvectors have sign ambiguity: (0.773, 0.634) and (−0.773, −0.634)
#      are the SAME direction — both are correct
#   ⚠️ Always STANDARDIZE before computing covariance for PCA
#
# ============================================================================
# WHERE EIGENVALUES APPEAR IN ML
# ============================================================================
#
#   PCA (Formula 08)                → eigenvalues of covariance matrix
#   Recommendation systems         → eigenvalues of user-item matrix
#   Google PageRank                → principal eigenvector of link graph
#   Spectral clustering            → eigenvalues of graph Laplacian
#   Kernel PCA                     → eigenvalues of kernel matrix
#   Deep learning (attention)      → eigen-decomposition of attention
#   Finance (risk factors)         → eigenvalues of covariance of returns
#   Quantum mechanics              → energy eigenvalues
#
#   Wherever there is a square matrix, eigenvalues tell the truth.
#
# ============================================================================
