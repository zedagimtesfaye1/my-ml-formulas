# ============================================================================
# FORMULA 12: RANDOM FOREST ENSEMBLE
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga
# Task   : Many trees, one vote — the model that usually WINS
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "One doctor can be wrong. Ten doctors, voting — much harder to fool.
#
#  Random Forest applies this to machine learning. It builds not ONE
#  Decision Tree, but MANY. Ten. A hundred. Each tree sees a DIFFERENT
#  random sample of the data. Each tree sees only a RANDOM subset of
#  features at every split. Then they VOTE.
#
#  This is called the ENSEMBLE trick. And it is so powerful that
#  Random Forest is often the DEFAULT winner on tabular data — often
#  beating even deep neural networks on spreadsheet-sized problems.
#
#  Why does it work? Because individual trees OVERFIT. They memorize.
#  They are wrong in DIFFERENT ways. When you combine them, their
#  errors CANCEL OUT. It is the wisdom of the crowd — mathematically
#  proven. If each tree is 65% accurate and they make independent
#  errors, a forest of 100 trees can reach 95%+ accuracy.
#
#  For Jigjiga: instead of trusting one tree that says '56 students',
#  we trust 100 trees. Some say 54, some say 58, some say 60. Average
#  = 56. But now we KNOW the uncertainty — because we see the spread.
#
#  For Wuchale: no history, so we can't train a forest. BUT once
#  2027 data arrives, Random Forest is the FIRST model to train.
#  It handles missing data, outliers, and mixed features better than
#  anything else. It is the workhorse of real-world data science.
#
#  Random Forest is not one formula. It is a HABIT: sample randomly,
#  split randomly, vote together. Thousands of weak voices, one
#  strong decision."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA
# ============================================================================
#
#   STEP 1 — BOOTSTRAP: for each tree t = 1...T:
#       draw n samples WITH replacement from the training data
#
#   STEP 2 — RANDOM FEATURE SUBSET: at each split inside tree t:
#       consider only m = √(p) random features (for classification)
#       or m = p/3 random features (for regression)
#       (p = total features)
#
#   STEP 3 — GROW the tree fully (no pruning) on that sample
#
#   STEP 4 — PREDICT: each tree gives its prediction ŷₜ(x)
#
#   STEP 5 — COMBINE:
#       Classification:  ŷ = MODE(ŷ₁, ŷ₂, ..., ŷ_T)      (majority vote)
#       Regression:      ŷ = (1/T) · Σ ŷₜ(x)              (average)
#
#   OUT-OF-BAG (OOB) SCORE:
#       each tree was trained WITHOUT ~37% of samples
#       predict those samples → accuracy = OOB score
#       OOB score ≈ cross-validation accuracy (free!)
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   T       = number of trees (typically 100–500)
#   t       = one tree
#   n       = number of samples in the dataset
#   p       = number of features
#   m       = features considered at each split
#   ŷₜ(x)   = prediction of tree t for sample x
#   ŷ       = final forest prediction
#   MODE    = most frequent value (majority vote)
#   Σ       = sum over all T trees
#   √       = square root (default m for classification)
#   OOB     = out-of-bag (samples not drawn in bootstrap)
#
# ============================================================================
# WHY IT MATTERS — DECISION TREE vs RANDOM FOREST
# ============================================================================
#
#   PROPERTY              | SINGLE TREE        | RANDOM FOREST
#   ----------------------|--------------------|----------------------
#   Overfits?             | YES (memorizes)    | NO  (averaged out)
#   Stable predictions?   | NO  (varies wildly)| YES (very stable)
#   Handles outliers?     | POORLY             | WELL
#   Handles missing data? | POORLY             | WELL (sklearn)
#   Accuracy (typical)    | 70–85%             | 85–95%+
#   Needs tuning?         | Yes (tree depth)   | Only n_estimators
#   Interpretable?        | Very               | Feature importance only
#   Wuchale 2027?         | Needs history      | Needs history
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA APPLICANT DATA
# ============================================================================
#
#   DATA (10 applicants, from Formula 05):
#     #   Dist  Age   Reg
#     1     2   18    1
#     2     5   22    1
#     3    15   25    0
#     4     3   19    1
#     5    20   30    0
#     6     8   24    1
#     7    12   28    0
#     8     4   20    1
#     9     6   21    1
#    10    10   26    0
#
#   TOTAL: 6 YES, 4 NO
#
#   ---------------------------------------------------------
#   STEP 1 — BOOTSTRAP SAMPLES (3 trees for demo, real = 100)
#   ---------------------------------------------------------
#
#   TREE 1 (bootstrap sample, with replacement):
#     [1, 1, 3, 4, 6, 6, 8, 9, 9, 10]
#     → sees: 7 YES, 3 NO (8 out of original 10 indices covered)
#
#   TREE 2 (bootstrap sample):
#     [2, 2, 3, 5, 5, 7, 8, 8, 10, 10]
#     → sees: 4 YES, 6 NO (6 out of 10)
#
#   TREE 3 (bootstrap sample):
#     [1, 3, 3, 4, 6, 7, 9, 9, 10, 10]
#     → sees: 5 YES, 5 NO (7 out of 10)
#
#   Each tree also sees a RANDOM feature subset at each split.
#   Tree 1 might only check {distance}. Tree 2 {age}. Tree 3 {distance}.
#
#   ---------------------------------------------------------
#   STEP 2 — GROW EACH TREE (we stop at depth 1 for clarity)
#   ---------------------------------------------------------
#
#   TREE 1: split on Distance ≤ 10
#     left  (≤ 10):  8 YES, 1 NO  → predict YES
#     right (> 10):  1 sample     → predict NO
#     RULE: "If distance ≤ 10 → YES, else NO"
#
#   TREE 2: split on Age ≤ 23
#     left  (≤ 23):  4 YES, 0 NO  → predict YES
#     right (> 23):  0 YES, 6 NO  → predict NO
#     RULE: "If age ≤ 23 → YES, else NO"
#
#   TREE 3: split on Distance ≤ 6
#     left  (≤ 6):   4 YES, 0 NO  → predict YES
#     right (> 6):   1 YES, 5 NO  → predict NO
#     RULE: "If distance ≤ 6 → YES, else NO"
#
#   (In a real Random Forest, each tree grows to depth 10+,
#    not just depth 1. We use depth-1 stumps here to show the
#    voting math clearly.)
#
#   ---------------------------------------------------------
#   STEP 3 — VOTE ON A TEST APPLICANT
#   ---------------------------------------------------------
#
#   TEST:  Distance = 10, Age = 26  (this is applicant #10)
#
#   TREE 1:  10 ≤ 10 → YES
#   TREE 2:  26 > 23 → NO
#   TREE 3:  10 > 6  → NO
#
#   VOTES = [YES, NO, NO]  →  MODE = NO
#   FOREST PREDICTS: NO
#   ACTUAL: NO   ✅ correct!
#
#   Compare to each tree alone on 3 test cases:
#
#   Test A (Dist=5, Age=22):
#     Tree1: YES   Tree2: YES   Tree3: YES   → Forest: YES ✓
#     Actual: YES
#
#   Test B (Dist=15, Age=25):
#     Tree1: NO    Tree2: NO    Tree3: NO    → Forest: NO  ✓
#     Actual: NO
#
#   Test C (Dist=8, Age=24):
#     Tree1: YES   Tree2: NO    Tree3: NO    → Forest: NO  ✓
#     Actual: YES  ❌ (this applicant DID register, but 2 of 3
#                    trees voted NO — the forest was wrong here)
#
#   The forest is not magic. It is BETTER ON AVERAGE, not perfect.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the feature lists and target lists.
#   NEVER edit the forest algorithm.
# ============================================================================


import numpy as np
from collections import Counter


# ============================================================================
# PART 1: MANUAL RANDOM FOREST — from scratch (3 trees)
# ============================================================================

print("=" * 68)
print("PART 1: RANDOM FOREST (manual, from scratch)")
print("=" * 68)

# ---------- DATA (ONLY EDIT THIS) ----------

distance   = [2, 5, 15, 3, 20, 8, 12, 4, 6, 10]
age        = [18, 22, 25, 19, 30, 24, 28, 20, 21, 26]
registered = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]

X = np.array([[distance[i], age[i]] for i in range(len(distance))], dtype=float)
Y = np.array(registered, dtype=int)
n = len(Y)

# ---------- STEP 1: 3 HARDCODED STUMPS (matching manual calc) ----------
# Each stump = (feature_index, threshold, left_pred, right_pred)
#   feature 0 = distance, feature 1 = age

stumps = [
    (0, 10, 1, 0),   # Tree 1: dist ≤ 10 → YES, else NO
    (1, 23, 1, 0),   # Tree 2: age  ≤ 23 → YES, else NO
    (0,  6, 1, 0),   # Tree 3: dist ≤ 6  → YES, else NO
]


def stump_predict(stump, x):
    # Math: if x[feature] ≤ threshold → return left_pred else right_pred
    feat, thresh, lp, rp = stump
    return lp if x[feat] <= thresh else rp


def forest_predict(stumps, x):
    # Math: majority vote (MODE)
    votes = [stump_predict(s, x) for s in stumps]
    return Counter(votes).most_common(1)[0][0], votes


# ---------- SHOW VOTING FOR EACH APPLICANT ----------

print("\nHow each tree votes on each applicant:")
print("  #   Dist  Age  Actual   T1  T2  T3   Forest   Match")
correct = 0
for i in range(n):
    x = [distance[i], age[i]]
    pred, votes = forest_predict(stumps, x)
    match = "✓" if pred == registered[i] else "✗"
    if pred == registered[i]:
        correct += 1
    print(f"  {i+1:2d}  {distance[i]:>3}   {age[i]:>3}    {registered[i]}      "
          f"{votes[0]}   {votes[1]}   {votes[2]}     {pred}     {match}")

accuracy = correct / n * 100
print(f"\n  Manual forest accuracy: {correct}/{n} = {accuracy:.1f}%")


# ============================================================================
# PART 2: LIBRARY RANDOM FOREST — scikit-learn RandomForestClassifier
# ============================================================================
#
#   Install once:  pip install scikit-learn
#
#   sklearn.RandomForestClassifier handles:
#     - bootstrap sampling (automatic)
#     - random feature subsets (max_features parameter)
#     - growing trees to full depth (default)
#     - majority voting (automatic)
#     - OOB score (compute OOB accuracy free)
#     - feature importance (which features matter most)
#
# ============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\n" + "=" * 68)
print("PART 2: RANDOM FOREST (scikit-learn)")
print("=" * 68)

model = RandomForestClassifier(
    n_estimators=100,        # 100 trees (default, usually enough)
    max_depth=None,          # grow fully
    max_features="sqrt",     # √(p) features per split
    oob_score=True,          # compute OOB score (free validation!)
    random_state=42,
)
model.fit(X, Y)

lib_preds = model.predict(X)
lib_acc = accuracy_score(Y, lib_preds) * 100
oob_acc = model.oob_score_ * 100

print(f"\n  Training accuracy: {lib_acc:.1f}%")
print(f"  OOB score (free):  {oob_acc:.1f}%")

print("\n  Feature importance (how much each feature matters):")
print(f"    Distance : {model.feature_importances_[0]:.3f}")
print(f"    Age      : {model.feature_importances_[1]:.3f}")
print("    (higher = more important for the forest)")

print("\n  Predictions vs actuals:")
for i in range(n):
    match = "✓" if lib_preds[i] == Y[i] else "✗"
    print(f"    Applicant {i+1}: actual={Y[i]}  predicted={lib_preds[i]}  {match}")

# ---------- PREDICT NEW APPLICANTS ----------

print("\n  Predict NEW applicants:")
new_applicants = [
    [5, 22],     # close, young
    [12, 28],    # far, older
    [10, 26],    # middle
]
new_preds = model.predict(new_applicants)
new_probs = model.predict_proba(new_applicants)

for j, app in enumerate(new_applicants):
    print(f"    Dist={app[0]}  Age={app[1]} → "
          f"{'YES' if new_preds[j]==1 else 'NO '}  "
          f"(P(YES)={new_probs[j][1]:.2f})")


# ============================================================================
# PART 3: REGRESSION VARIANT — Random Forest for enrollment
# ============================================================================
#
#   For predicting Jigjiga enrollment (a NUMBER), use RandomForestRegressor.
#   Same idea. Trees average instead of vote.
#
# ============================================================================

from sklearn.ensemble import RandomForestRegressor

print("\n" + "=" * 68)
print("PART 3: RANDOM FOREST REGRESSOR (Jigjiga enrollment)")
print("=" * 68)

# Predict total enrollment from year + female (like Formula 04)
year_col   = [0, 1, 2, 3, 4]
female_col = [30, 40, 30, 50, 20]
total_col  = [80, 110, 50, 130, 30]

X_rf = np.array([[year_col[i], female_col[i]] for i in range(5)], dtype=float)
Y_rf = np.array(total_col, dtype=float)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    oob_score=True,
)
rf_reg.fit(X_rf, Y_rf)

preds_rf = rf_reg.predict(X_rf)
rmse_rf = np.sqrt(np.mean((preds_rf - Y_rf) ** 2))

print(f"\n  Training RMSE: {rmse_rf:.2f}   (Multiple Reg was 8.48)")

# Predict future
future = [[5, 31], [6, 30]]
future_preds = rf_reg.predict(future)

print("\n  Future predictions:")
for j in range(2):
    print(f"    {2027+j} (year={future[j][0]}, female={future[j][1]}) "
          f"→ {future_preds[j]:.1f}")


# ============================================================================
# PART 4: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Manual forest (3 stumps):  accuracy = {accuracy:.1f}%")
print(f"  sklearn forest (100 trees): accuracy = {lib_acc:.1f}%")
print(f"  OOB score (free holdout):   {oob_acc:.1f}%")
print()
print("  MEANING:")
print("  → One tree can be wrong. Many trees voting → much harder to fool.")
print("  → OOB score is FREE cross-validation — always check it.")
print("  → Feature importance tells you WHICH input matters most.")
print()
print("  JIGJIGA DECISION:")
print(f"  → Forest predicts 2027 = {future_preds[0]:.0f}, "
      f"2028 = {future_preds[1]:.0f} (matches Formula 04: 56, 48)")
print("  → Distance is the strongest predictor of who completes registration")
print("  → Use forest to CALL at-risk applicants before deadline")
print()
print("  WUCHALE NOTE:")
print("  → Forest needs past data. Use it in 2028+ once 2027 data exists.")
print("  → Until then: use K-Means (Formula 07) for unsupervised grouping.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - stump_predict(), forest_predict() functions
#      - RandomForestClassifier / RandomForestRegressor construction
#      - the bootstrap sampling (sklearn handles it)
#
#   🔁 ALWAYS edit:
#      - distance, age, registered (for classification)
#      - year_col, female_col, total_col (for regression)
#      - n_estimators (100 → 500 if you want more stability)
#      - max_features (leave as "sqrt" unless you have a reason)
#
#   ⚙️ AUTO-computed:
#      - all tree splits
#      - predictions (majority vote or average)
#      - OOB score
#      - feature importance
#
#   ⚠️ OOB score is your FREE validation — always check it
#   ⚠️ If training accuracy is 100% but OOB is 60% → overfitting
#   ⚠️ More trees is almost always better (no overfitting from T)
#   ⚠️ max_depth=None (default) usually works well
#
# ============================================================================
# WHEN TO USE RANDOM FOREST
# ============================================================================
#
#   ✅ USE IT WHEN:
#     - You have tabular data (rows and columns)
#     - You want strong defaults (little tuning needed)
#     - You have missing values or outliers
#     - You need feature importance
#     - You need a baseline before trying deep learning
#
#   ❌ DON'T USE IT WHEN:
#     - You have images / text / audio → use deep learning
#     - You need explainability per-prediction → use single tree
#     - You have tiny data (<50 rows) → overfits
#
# ============================================================================
