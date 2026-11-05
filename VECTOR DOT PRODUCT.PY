# ============================================================================
# FORMULA 10: VECTOR DOT PRODUCT  —  a · b
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga + Wuchale
# Task   : The single operation that powers every formula you've learned
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Every formula you have learned so far — Linear Regression, Multiple
#  Regression, Logistic Regression, Neural Network, PCA — all of them
#  use the SAME operation, over and over. It is called the DOT PRODUCT.
#
#  It is simple. You take two lists of numbers. Multiply them position
#  by position. Add the results. That is it. That is the dot product.
#
#  Two lists, five numbers each? Five multiplications, four additions.
#  Two lists, thirty numbers each? Thirty multiplications. Still just
#  one number out.
#
#  And yet — this one line is the HEART of artificial intelligence.
#    - Every prediction of Linear Regression      → one dot product.
#    - Every forward pass of a Neural Network     → one dot product.
#    - Every covariance cell in PCA               → one dot product.
#    - Every distance in K-Means                  → one dot product.
#    - Every 'similarity' in any AI               → one dot product.
#
#  When ChatGPT processes a sentence, it computes THOUSANDS of dot
#  products per word. When a self-driving car sees an image, it is
#  just a sea of dot products. When AlphaFold predicts a protein,
#  it is billions of dot products.
#
#  For Wuchale: we want to know how 'close' one student is to another.
#  That is a dot product. We want to predict their registration score.
#  Dot product. We want to cluster them. Dot product.
#
#  If machine learning is a machine, the dot product is the BOLT.
#  Millions of bolts, holding the whole thing together. Today you
#  will see the bolt. Once you see it, you will see it everywhere."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA — 3 EQUIVALENT FORMS
# ============================================================================
#
#   FORM 1 — ALGEBRAIC (what computers actually do):
#
#       a · b = a₁b₁ + a₂b₂ + ... + aₚbₚ
#
#       In sigma notation:
#       a · b = Σᵢ (aᵢ · bᵢ)      for i = 1 to p
#
#   FORM 2 — GEOMETRIC (what it MEANS):
#
#       a · b = |a| · |b| · cos(θ)
#
#       where |a| = length of a = √(a₁² + a₂² + ... + aₚ²)
#             θ  = angle between the two vectors
#
#   FORM 3 — MATRIX (how it appears in ML):
#
#       For X (n × p) and W (p × 1):
#       prediction[i] = X[i] · W = Σⱼ X[i][j] · W[j]
#
#       (each row of X dotted with the weight vector W)
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   a, b    = vectors of the same length (lists of numbers)
#   aᵢ, bᵢ  = the i-th element of each vector
#   p       = number of elements (length of the vectors)
#   ·       = dot product (NOT regular multiplication)
#   Σ       = sum over all positions
#   |a|     = length (magnitude) of vector a
#   θ       = angle between the vectors (in radians or degrees)
#   cos     = cosine of the angle
#   X[i]    = row i of matrix X (a vector)
#   W       = weight vector (single column)
#   X @ W   = matrix-times-vector = one dot product per row
#
# ============================================================================
# WHY IT MATTERS — IT IS INSIDE EVERY FORMULA YOU LEARNED
# ============================================================================
#
#   FORMULA 01 (Linear Regression):
#       Y = W·X + B
#       → For 2 features: Y = W₁·X₁ + W₂·X₂ + B  =  (W·X)  +  B
#       → the (W·X) part IS a dot product.
#
#   FORMULA 04 (Multiple Regression):
#       Y = W₁X₁ + W₂X₂ + ... + B
#       → Same. The whole linear part is ONE dot product.
#
#   FORMULA 05 (Logistic Regression):
#       z  = W·X + B          ← dot product
#       ŷ  = sigmoid(z)
#
#   FORMULA 07 (K-Means):
#       distance² = (x−c)·(x−c)   ← dot product of a vector with itself
#
#   FORMULA 08 (PCA):
#       Z = Xc · W            ← each row of Xc dotted with W
#
#   FORMULA 09 (Neural Network):
#       z_h = X·W_h + b_h     ← dot product
#       z_o = a_h·W_o + b_o   ← dot product
#
#   EVERY formula is dot products all the way down.
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA / WUCHALE DATA (4 EXAMPLES)
# ============================================================================
#
#   EXAMPLE 1 — LINEAR REGRESSION PREDICTION FOR 2027
#   ---------------------------------------------
#   From Formula 04, the learned coefficients are:
#       W = [-4.63, 3.37]     (W₁ = year, W₂ = female)
#       B = -25.41
#
#   For 2027:  X = [Year, Female] = [5, 31]
#
#   STEP 1 — dot product  W · X :
#       (-4.63) · 5   = -23.15
#       (3.37) · 31   = 104.47
#       SUM           =  81.32
#
#   STEP 2 — add B:
#       Y = 81.32 + (-25.41) = 55.91
#
#   → That is the SAME 55.9 prediction from Formula 04.
#   → In ONE dot product + ONE addition. No magic.
#
#   EXAMPLE 2 — NEURAL NETWORK HIDDEN LAYER
#   ---------------------------------------------
#   From Formula 09, hidden weights:
#       W_h = [0.5, 0.5],  b_h = 0.0
#
#   For sample  Year=1, Female=40:  X = [1, 40]
#
#   STEP 1 — dot product  W_h · X :
#       0.5 · 1   = 0.5
#       0.5 · 40  = 20.0
#       SUM       = 20.5
#
#   STEP 2 — add bias and apply ReLU:
#       z_h = 20.5 + 0 = 20.5
#       a_h = ReLU(20.5) = 20.5
#
#   → Same 20.5 you saw in Formula 09. Because it IS a dot product.
#
#   EXAMPLE 3 — K-MEANS DISTANCE (Wuchale)
#   ---------------------------------------------
#   Distance between student P1 = (2, 16) and centroid c₁ = (3.33, 16.0):
#
#   STEP 1 — difference vector  d = P1 − c₁ :
#       d = (2 − 3.33, 16 − 16.0) = (-1.33, 0.0)
#
#   STEP 2 — dot product of d with itself:
#       d · d = (-1.33)·(-1.33) + (0.0)·(0.0)
#             = 1.7689 + 0
#             = 1.7689
#
#   STEP 3 — distance = √(d · d) :
#       distance = √1.7689 = 1.33
#
#   → That is the Euclidean distance — built from ONE dot product.
#
#   EXAMPLE 4 — PCA PROJECTION (Wuchale)
#   ---------------------------------------------
#   From Formula 08, PC1 direction:
#       v = [0.773, 0.634]
#
#   For Student A centered = (-4, -3.6):
#
#   STEP 1 — dot product  xc · v :
#       (-4) · 0.773     = -3.092
#       (-3.6) · 0.634   = -2.282
#       SUM              = -5.374
#
#   → That is the Z = -5.374 for Student A. One dot product.
#
#   CONCLUSION:
#       Every "prediction" you've ever seen is one dot product
#       (plus a bias, plus maybe a nonlinear squish).
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the vectors a and b (same length).
#   NEVER edit the dot_product function.
# ============================================================================


import math
import numpy as np


# ============================================================================
# PART 1: MANUAL DOT PRODUCT — from scratch
# ============================================================================

print("=" * 68)
print("PART 1: VECTOR DOT PRODUCT (manual, from scratch)")
print("=" * 68)


def dot_product(a, b):
    # Math:  a · b = Σ (aᵢ · bᵢ)
    if len(a) != len(b):
        raise ValueError("Vectors must have the SAME length")
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def vector_length(a):
    # Math:  |a| = √(a · a)
    return math.sqrt(dot_product(a, a))


def angle_between(a, b):
    # Math:  cos(θ) = (a·b) / (|a|·|b|)
    cos_theta = dot_product(a, b) / (vector_length(a) * vector_length(b))
    cos_theta = max(-1.0, min(1.0, cos_theta))   # clamp for numerical safety
    return math.degrees(math.acos(cos_theta))


# ---------- EXAMPLE 1: JIGJIGA 2027 PREDICTION ----------

print("\nEXAMPLE 1: Linear Regression prediction for 2027")
W = [-4.63, 3.37]                 # learned coefficients from Formula 04
X = [5, 31]                       # 2027: year=5, female=31
B = -25.41

print(f"  W     = {W}")
print(f"  X     = {X}")
print(f"  W · X = {dot_product(W, X):+.2f}")
print(f"  + B   = {dot_product(W, X) + B:.2f} students in 2027")


# ---------- EXAMPLE 2: NEURAL NETWORK HIDDEN LAYER ----------

print("\nEXAMPLE 2: Neural Network hidden layer")
W_h = [0.5, 0.5]
X2  = [1, 40]                     # Year=1, Female=40

print(f"  W_h   = {W_h}")
print(f"  X     = {X2}")
print(f"  W_h·X = {dot_product(W_h, X2):.2f}")
print(f"  ReLU  = {max(0.0, dot_product(W_h, X2)):.2f}")


# ---------- EXAMPLE 3: K-MEANS DISTANCE (Wuchale) ----------

print("\nEXAMPLE 3: K-Means distance (Wuchale)")
P1 = [2, 16]
c1 = [3.33, 16.0]
diff = [P1[0] - c1[0], P1[1] - c1[1]]

print(f"  P1 − c1 = {diff}")
print(f"  d · d   = {dot_product(diff, diff):.4f}")
print(f"  distance = √(d·d) = {math.sqrt(dot_product(diff, diff)):.2f}")


# ---------- EXAMPLE 4: PCA PROJECTION (Wuchale) ----------

print("\nEXAMPLE 4: PCA projection (Wuchale)")
xc = [-4.0, -3.6]
v  = [0.773, 0.634]
print(f"  xc      = {xc}")
print(f"  v (PC1) = {v}")
print(f"  xc · v  = {dot_product(xc, v):+.3f}   ← Z for Student A")


# ---------- BONUS: ANGLE BETWEEN TWO VECTORS ----------

print("\nBONUS: angle between two vectors (the geometric meaning)")
a = [1, 0]
b = [1, 1]
print(f"  a = {a}   b = {b}")
print(f"  |a| = {vector_length(a):.4f}")
print(f"  |b| = {vector_length(b):.4f}")
print(f"  a · b = {dot_product(a, b):.4f}")
print(f"  cos(θ) = {dot_product(a,b)/(vector_length(a)*vector_length(b)):.4f}")
print(f"  θ = {angle_between(a, b):.2f}°")


# ============================================================================
# PART 2: LIBRARY VERSION — numpy (production code)
# ============================================================================
#
#   numpy provides np.dot(a, b) and the @ operator.
#   Both do the SAME thing — one is a function, one is an operator.
#   Same performance. Same result. Use @ for readability.
#
# ============================================================================

print("\n" + "=" * 68)
print("PART 2: VECTOR DOT PRODUCT (numpy library)")
print("=" * 68)

# ---------- EXAMPLE 1: numpy dot ----------

W_np = np.array([-4.63, 3.37])
X_np = np.array([5, 31])
B_np = -25.41

result_1 = np.dot(W_np, X_np) + B_np
print(f"\nEXAMPLE 1: np.dot(W, X) + B = {result_1:+.2f} students in 2027")

# Same result using @ operator
result_1b = W_np @ X_np + B_np
print(f"           W @ X + B       = {result_1b:+.2f}   (same thing)")


# ---------- EXAMPLE 2: batch dot products (matrix @ vector) ----------
# This is how ALL of machine learning actually runs.
# X is a MATRIX (5 samples × 2 features). W is a VECTOR (2 weights).
# X @ W = 5 dot products — one per row — in ONE line.

print("\nEXAMPLE 2: predict ALL years at once with one line")
year_col   = [0, 1, 2, 3, 4]
female_col = [30, 40, 30, 50, 20]

X_matrix = np.array([[year_col[i], female_col[i]] for i in range(5)], dtype=float)
W_vector = np.array([-4.63, 3.37])

predictions = X_matrix @ W_vector + (-25.41)

print(f"  X_matrix shape = {X_matrix.shape}   (5 rows, 2 features)")
print(f"  W_vector shape = {W_vector.shape}   (2 weights)")
print(f"  X @ W + B     → {predictions.shape}   (5 predictions)")
print("\n  Predictions vs actuals:")
actual = [80, 110, 50, 130, 30]
for i in range(5):
    print(f"    Year={year_col[i]}  Female={female_col[i]:>2}  "
          f"Actual={actual[i]:>3}  Predicted={predictions[i]:7.2f}")


# ---------- EXAMPLE 3: how XᵀX is built from dot products ----------
# Each cell of XᵀX is the dot product of two COLUMNS of X.

print("\nEXAMPLE 3: each XᵀX cell is a dot product of two columns")

col_year   = X_matrix[:, 0]
col_female = X_matrix[:, 1]

cell_00 = col_year @ col_year           # year · year
cell_01 = col_year @ col_female         # year · female
cell_11 = col_female @ col_female       # female · female

print(f"  XᵀX[0][0] = year · year       = {cell_00:.2f}")
print(f"  XᵀX[0][1] = year · female     = {cell_01:.2f}")
print(f"  XᵀX[1][1] = female · female   = {cell_11:.2f}")
print("  → every cell in the covariance / XᵀX matrix is ONE dot product")


# ---------- EXAMPLE 4: neural network forward pass with numpy ----------

print("\nEXAMPLE 4: neural network forward pass with numpy")

W_h_np = np.array([0.5, 0.5])
X_nn   = np.array([[0, 30], [1, 40], [2, 30], [3, 50], [4, 20]], dtype=float)
b_h_np = 0.0

z_h = X_nn @ W_h_np + b_h_np                 # ← ALL dot products at once
a_h = np.maximum(0, z_h)                     # ← ReLU in one line

print(f"  z_h = {z_h}")
print(f"  a_h = {a_h}   (ReLU applied)")


# ============================================================================
# PART 3: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print("  Manual  a · b      = Σ(aᵢ · bᵢ)")
print("  Manual  |a|        = √(a · a)")
print("  numpy   np.dot(a,b) = same thing, in C-speed")
print("  numpy   a @ b       = same thing, as operator")
print()
print("  Where the dot product lives in your formulas:")
print("    Formula 01:  Y = W · X + B")
print("    Formula 04:  Y = W₁X₁ + W₂X₂ + B      =  W · X + B")
print("    Formula 05:  z = W · X + B   → sigmoid")
print("    Formula 07:  distance² = (x−c) · (x−c)")
print("    Formula 08:  Z = Xc · W")
print("    Formula 09:  z_h = X · W_h   ,   z_o = a_h · W_o")
print()
print("  DECISION — what this means for Jigjiga / Wuchale:")
print("    → One line — X @ W + B — does all the work.")
print("    → All 9 formulas are just dot products with different wrappers.")
print("    → Once you master this, you master the atom of AI.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - dot_product(), vector_length(), angle_between()
#      - any np.dot / @ line
#
#   🔁 ALWAYS edit:
#      - the vectors (a, b) — same length required
#      - W, X, B in the examples to match your data
#
#   ⚠️ Both vectors MUST be the same length
#   ⚠️ For matrices: (n × p) @ (p × 1) → (n × 1)  — inner dimensions must match
#
# ============================================================================
# GOLDEN RULE
# ============================================================================
#
#   In machine learning, when in doubt about what a formula is doing —
#   it is almost always a dot product.
#
# ============================================================================
