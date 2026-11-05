# ============================================================================
# FORMULA 14: SUPPORT VECTOR MACHINE (SVM)
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga
# Task   : Classification by MARGIN — the widest street between two groups
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Suppose I give you two groups of dots on a page — YES students and NO
#  students. Your job: draw a line that separates them.
#
#  Here is the problem. You can draw INFINITE lines that separate them.
#  Which one is right?
#
#  SVM answers with one idea: choose the line with the WIDEST MARGIN.
#  The street between the two groups should be as wide as possible.
#  Why? Because a wide street is robust — a new student appearing near
#  the boundary is less likely to fall on the wrong side.
#
#  And the students closest to that street? They are the ones the line
#  is really built around. Nobody else matters. These are the SUPPORT
#  VECTORS. The model is NAMED after them. Remove a hundred students far
#  from the boundary — nothing changes. Remove ONE support vector — the
#  line moves. That is the power of margins.
#
#  For Jigjiga: which applicants will complete registration? We have
#  10 applicants. SVM draws the widest safe zone between 'will register'
#  and 'will not register' — based on distance and age. It gives us
#  a line. And it tells us WHICH applicants are the closest cases —
#  the borderline ones. Those are the ones we should CALL before the
#  deadline. SVM doesn't just classify — it tells us who is uncertain.
#
#  For Wuchale: no history, so we cannot train SVM yet. But once 2027
#  data arrives, SVM is one of the strongest small-data classifiers.
#  It works even with 30–50 samples, where neural networks fail.
#
#  SVM has one more trick: THE KERNEL. If groups cannot be separated
#  by a straight line, SVM bends the space itself. It lifts the data
#  into higher dimensions where a straight line IS possible. This is
#  the kernel trick — one of the most beautiful ideas in all of math.
#
#  For Wuchale 30,000 students: maybe the true boundary is curved.
#  SVM with an RBF kernel will find it. Linear models cannot."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA
# ============================================================================
#
#   GOAL — Find the hyperplane  w·x + b = 0  that maximizes the margin:
#
#       maximize     2 / ||w||
#       subject to   yᵢ · (w·xᵢ + b) ≥ 1     for all i
#
#   where yᵢ ∈ {−1, +1} (labels must be −1 and +1, NOT 0 and 1)
#
#   EQUIVALENT MINIMIZATION FORM (what solvers actually run):
#
#       minimize     (1/2) · ||w||²
#       subject to   yᵢ · (w·xᵢ + b) ≥ 1     for all i
#
#   SOFT MARGIN FORM (allow some errors — used in real data):
#
#       minimize     (1/2)·||w||²  +  C · Σ ξᵢ
#       subject to   yᵢ·(w·xᵢ + b) ≥ 1 − ξᵢ ,   ξᵢ ≥ 0
#
#   DECISION RULE (predict class for new sample x):
#
#       ŷ = sign( w·x + b )
#       → +1 if w·x + b > 0     (class YES)
#       → −1 if w·x + b < 0     (class NO)
#
#   KERNEL TRICK (for non-linear boundaries):
#
#       Replace  xᵢ·xⱼ  with  K(xᵢ, xⱼ):
#       Linear kernel:  K = xᵢ·xⱼ
#       Polynomial:     K = ( xᵢ·xⱼ + 1 )^d
#       RBF (Gaussian): K = exp( −γ · ||xᵢ − xⱼ||² )
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   w        = weight vector (direction of the hyperplane)
#   b        = bias (offset)
#   x        = input features (one row)
#   xᵢ, yᵢ   = i-th sample and its label
#   yᵢ       = label, ∈ {−1, +1}
#   ||w||    = length of w = √(w · w)
#   C        = regularization (large C = hard margin, small C = soft)
#   ξᵢ       = slack variable (how much i-th sample violates margin)
#   K        = kernel function
#   γ (gamma)= RBF width parameter
#   d        = polynomial degree
#   ·        = dot product (Formula 10!)
#   sign()   = +1 if positive, −1 if negative
#
# ============================================================================
# WHY IT MATTERS — LINEAR SEPARATORS ARE NOT ENOUGH
# ============================================================================
#
#   METHOD                 | IDEA                         | HANDLES
#   -----------------------|------------------------------|----------------
#   Logistic Regression    | Probability boundary         | Linear
#   SVM (linear)           | WIDEST margin line           | Linear
#   SVM (RBF kernel)       | Bends space, wide margin     | NON-LINEAR
#   Random Forest          | Many trees vote              | Non-linear
#   Neural Net             | Deep layers                  | Any
#
#   SVM is the KING of small-data, high-dimension classification.
#   Works well with 30 samples. Neural nets fail there.
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA APPLICANT DATA
# ============================================================================
#
#   DATA (10 applicants, simplified to 1 feature — Distance — for clarity):
#
#     #   Distance (km)   Registered   yᵢ
#     1       2             1          +1
#     2       3             1          +1
#     3       4             1          +1
#     4       5             1          +1
#     5       6             1          +1
#     6       12            0          −1
#     7       15            0          −1
#     8       18            0          −1
#     9       20            0          −1
#     10      25            0          −1
#
#   (For a 1-feature SVM, the hyperplane is just a THRESHOLD value
#    b, not a full weight vector — but the math is the same.)
#
#   ---------------------------------------------------------
#   SUPPORT VECTORS
#   ---------------------------------------------------------
#   The closest points to the boundary are:
#     From YES side:  x = 6   (applicant #5)
#     From NO  side:  x = 12  (applicant #6)
#
#   The widest margin between them:
#     MIDPOINT = (6 + 12) / 2 = 9
#     MARGIN   = (12 − 6) / 2 = 3  (half-width)
#
#   The DECISION BOUNDARY is at x = 9.
#   Decision rule: if Distance < 9 → YES ; if Distance > 9 → NO
#
#   SUPPORT VECTORS are ONLY these two: x = 6 and x = 12.
#   The other 8 students could move around within their side
#   and the boundary WOULD NOT CHANGE. That is what makes SVM
#   robust — it is built on the borderline cases, not the crowd.
#
#   ---------------------------------------------------------
#   SOFT MARGIN — introducing C
#   ---------------------------------------------------------
#   If the data is NOT perfectly separable (like Jigjiga with
#   two features distance + age, where the classes overlap),
#   we allow violations. Each violation costs ξᵢ. Total cost
#   is multiplied by C. Larger C → fewer violations allowed
#   (stricter), smaller C → more violations tolerated (softer).
#
#   Example with two features for applicant #10:
#     Distance = 10, Age = 26, Actual = NO (y = −1)
#     If the boundary gives w·x + b = +0.3, then:
#       y·(w·x+b) = (−1)·(+0.3) = −0.3 < 1  → violation
#       ξᵢ = 1 − (−0.3) = 1.3  (how much it violates)
#     This contributes C·1.3 to the loss — the model pays for it.
#
#   ---------------------------------------------------------
#   KERNEL TRICK (for non-linear data)
#   ---------------------------------------------------------
#   Imagine applicants where YES/NO depends on distance in a CURVED
#   way: near students YES, far students NO, but VERY far students
#   YES again (they enroll in satellite campuses).
#
#   A straight line cannot separate this. RBF kernel can.
#
#   RBF:  K(xᵢ, xⱼ) = exp( −γ · ||xᵢ − xⱼ||² )
#
#   For sample 1 (dist=2) and sample 10 (dist=25), γ = 0.1:
#     ||x₁ − x₁₀||² = (2−25)² = 529
#     K = exp(−0.1 · 529) = exp(−52.9) ≈ 1.2 × 10⁻²³  (nearly zero)
#
#   Two far-apart points are almost unrelated under RBF.
#   This is how SVM bends the space — it uses similarity to
#   reorganize distance.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit distance, age, registered lists.
#   NEVER edit the SVM math or sklearn call.
#   Labels MUST be 0/1 in sklearn, but math uses −1/+1 internally.
# ============================================================================


import numpy as np
import math


# ============================================================================
# PART 1: MANUAL SVM — 1D distance example (the whole math visible)
# ============================================================================

print("=" * 68)
print("PART 1: SVM (manual, 1D distance example)")
print("=" * 68)

# ---------- DATA (ONLY EDIT THIS FOR MANUAL DEMO) ----------

distance   = [2, 3, 4, 5, 6, 12, 15, 18, 20, 25]
registered = [1, 1, 1, 1, 1, 0,  0,  0,  0,  0]

# Convert labels to +1 / −1 (SVM convention)
y_svm = [1 if r == 1 else -1 for r in registered]

# ---------- STEP 1: FIND THE SUPPORT VECTORS ----------
# Support vectors = the closest YES-side and NO-side points

yes_side = [distance[i] for i in range(len(distance)) if y_svm[i] == +1]
no_side  = [distance[i] for i in range(len(distance)) if y_svm[i] == -1]

sv_yes = max(yes_side)       # closest YES to boundary
sv_no  = min(no_side)        # closest NO  to boundary

print(f"\nSupport vector on YES side:  distance = {sv_yes}")
print(f"Support vector on NO  side:  distance = {sv_no}")

# ---------- STEP 2: THE BOUNDARY (midpoint) ----------
# Math:  boundary = (sv_yes + sv_no) / 2

boundary = (sv_yes + sv_no) / 2
margin   = (sv_no - sv_yes) / 2

print(f"\nDecision boundary = ({sv_yes} + {sv_no}) / 2 = {boundary}")
print(f"Margin (half-width) = {margin}")

# ---------- STEP 3: DECISION RULE ----------
# Math:  if distance < boundary → YES ; else → NO

def svm_predict_1d(x):
    return 1 if x < boundary else 0

print("\nDecision rule: if Distance < 9 → YES, else NO")

correct = 0
print("\nApplicant  Distance  Actual  Predicted  Match")
for i in range(len(distance)):
    pred = svm_predict_1d(distance[i])
    ok   = "✓" if pred == registered[i] else "✗"
    if pred == registered[i]:
        correct += 1
    print(f"  {i+1:>2}        {distance[i]:>3}      {registered[i]}        {pred}       {ok}")

print(f"\n  Manual SVM accuracy: {correct}/{len(distance)} "
      f"= {correct/len(distance)*100:.1f}%")

# ---------- STEP 4: RBF KERNEL EXAMPLE ----------
# Math:  K(xᵢ, xⱼ) = exp( −γ · ||xᵢ − xⱼ||² )

print("\nRBF kernel values (γ = 0.1):")
gamma = 0.1
pairs = [(2, 2), (2, 6), (2, 12), (2, 25)]
for a, b in pairs:
    d2 = (a - b) ** 2
    K  = math.exp(-gamma * d2)
    print(f"  K({a:>2}, {b:>2}) = exp(−0.1·{d2:>4}) = {K:.6f}")


# ============================================================================
# PART 2: LIBRARY SVM — scikit-learn SVC (linear + RBF)
# ============================================================================
#
#   SVC = Support Vector Classifier.
#   Linear kernel → straight boundary.
#   RBF kernel    → curved boundary (kernel trick).
#
# ============================================================================

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 68)
print("PART 2: SVM (scikit-learn, 2-feature real data)")
print("=" * 68)

# ---------- DATA (ONLY EDIT THIS) ----------
# Same Jigjiga registrar applicants — 2 features

dist_arr  = [2, 5, 15, 3, 20, 8, 12, 4, 6, 10]
age_arr   = [18, 22, 25, 19, 30, 24, 28, 20, 21, 26]
reg_arr   = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]

X_svm = np.array([[dist_arr[i], age_arr[i]] for i in range(len(dist_arr))], dtype=float)
Y_svm = np.array(reg_arr, dtype=int)      # sklearn uses 0/1, not −1/+1

# ---------- STANDARDIZE (SVM is distance-based, so scaling matters) ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_svm)

# ---------- MODEL 1: LINEAR KERNEL ----------

print("\nMODEL 1: Linear kernel (straight boundary)")
svm_lin = SVC(kernel="linear", C=1.0)
svm_lin.fit(X_scaled, Y_svm)
pred_lin = svm_lin.predict(X_scaled)
acc_lin = accuracy_score(Y_svm, pred_lin) * 100

print(f"  Accuracy: {acc_lin:.1f}%")
print(f"  Support vectors: {len(svm_lin.support_)} "
      f"(only these define the boundary!)")
print(f"  Support vector indices: {svm_lin.support_}")

# ---------- MODEL 2: RBF KERNEL ----------

print("\nMODEL 2: RBF kernel (curved boundary)")
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_rbf.fit(X_scaled, Y_svm)
pred_rbf = svm_rbf.predict(X_scaled)
acc_rbf = accuracy_score(Y_svm, pred_rbf) * 100

print(f"  Accuracy: {acc_rbf:.1f}%")
print(f"  Support vectors: {len(svm_rbf.support_)}")

# ---------- SHOW WHICH ARE SUPPORT VECTORS ----------

print("\nWhich applicants are SUPPORT VECTORS (borderline)?")
for i in range(len(dist_arr)):
    marker = "  ← borderline" if i in svm_rbf.support_ else ""
    print(f"  Applicant {i+1}: dist={dist_arr[i]:>2} age={age_arr[i]:>2} "
          f"actual={reg_arr[i]}  pred={pred_rbf[i]}{marker}")

# ---------- PREDICT NEW APPLICANTS ----------

print("\nPredict 3 NEW applicants (RBF model):")
new_applicants = [[5, 22], [12, 28], [10, 26]]
new_scaled = scaler.transform(new_applicants)
new_pred   = svm_rbf.predict(new_scaled)
new_prob   = svm_rbf.decision_function(new_scaled)   # signed distance

for j in range(3):
    d, a = new_applicants[j]
    sign = "YES" if new_pred[j] == 1 else "NO "
    print(f"  Dist={d:>2} Age={a} → {sign}   "
          f"(signed distance = {new_prob[j]:+.2f})")


# ============================================================================
# PART 3: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Manual (1D distance): accuracy = {correct/len(distance)*100:.1f}%")
print(f"  Linear kernel:        accuracy = {acc_lin:.1f}%")
print(f"  RBF kernel:           accuracy = {acc_rbf:.1f}%")
print()
print("  MEANING:")
print("  → SVM builds the WIDEST safe street between classes.")
print("  → Only the borderline samples (support vectors) define the line.")
print("  → Few support vectors → robust model, not overfit.")
print("  → Kernel trick (RBF) bends the space to separate non-linear groups.")
print()
print("  JIGJIGA DECISION:")
print("  → Applicant 6 (dist=8, age=24) is a support vector — borderline.")
print("  → Those borderline applicants are the ones to CALL before deadline.")
print("  → Everything else, the model is confident. Don't waste calls on them.")
print()
print("  WUCHALE NOTE:")
print("  → SVM needs labeled history. Use in 2028+ after 2027 data exists.")
print("  → For 30,000 students with no labels, use K-Means (Formula 07).")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - the boundary/margin math in Part 1
#      - svm_predict_1d
#      - SVC(kernel=..., C=..., gamma=...)
#      - StandardScaler() fitting
#
#   🔁 ALWAYS edit:
#      - dist_arr, age_arr, reg_arr
#      - kernel ("linear" vs "rbf" vs "poly")
#      - C (regularization — higher C = stricter margin)
#      - gamma (RBF width — larger γ = more complex boundary)
#
#   ⚙️ AUTO-computed:
#      - support vectors
#      - w, b (from the fitted model)
#      - predictions and signed distances
#
#   ⚠️ Always STANDARDIZE features before SVM — it is distance-based
#   ⚠️ Labels must be 0/1 for sklearn (not −1/+1)
#   ⚠️ Large C + large γ = overfitting risk
#   ⚠️ Small C + small γ = underfitting (too flat)
#
# ============================================================================
# WHEN TO USE SVM
# ============================================================================
#
#   ✅ USE IT WHEN:
#     - Small to medium data (10 – 10,000 rows)
#     - High dimension features (text, gene data)
#     - Classes are separable (or close to it)
#     - You want a robust, low-variance classifier
#
#   ❌ DON'T USE IT WHEN:
#     - Very large datasets (> 1M rows — slow training)
#     - You need probability estimates (SVM needs extra calibration)
#     - Data is very noisy with heavy overlap
#     - You have no labels → use K-Means instead
#
# ============================================================================
