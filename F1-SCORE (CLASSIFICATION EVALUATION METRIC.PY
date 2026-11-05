# ============================================================================
# FORMULA 06: F1-SCORE  (EVALUATION METRIC FOR CLASSIFICATION)
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga, Ethiopia
# Task   : Measure how GOOD a Yes/No classifier really is
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Accuracy is a liar. Let me prove it.
#
#  Imagine 100 applicants. 95 will NOT register. 5 WILL register.
#  Now imagine a lazy model that says NO to every single person.
#  It gets 95 right out of 100. That is 95% accuracy!
#  Sounds amazing — but the model NEVER catches a single student
#  who would register. It is useless.
#
#  ACCURACY FAILED. Because accuracy does not care WHERE the model
#  is right or wrong. It just counts.
#
#  F1-SCORE is different. It punishes the model that misses
#  the 5 real students. It asks TWO questions:
#
#    1. Of everyone the model said YES to — how many really
#       were YES?       This is PRECISION.
#
#    2. Of everyone who really was YES — how many did the
#       model catch?    This is RECALL.
#
#  A good model needs BOTH. F1-SCORE combines them into ONE
#  number — the HARMONIC MEAN. If either Precision OR Recall
#  is low, F1 collapses. It cannot be gamed.
#
#  For Jigjiga: our model is 71.4% accurate by F1. Not perfect.
#  It catches 5 of 8 future students — misses 3. It also flags
#  1 who won't register. F1 tells us exactly where to improve:
#  CALL the ones we missed before the deadline.
#
#  F1 is the honest truth-teller of machine learning."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA
# ============================================================================
#
#   STEP 1 — Confusion Matrix (counts from comparing actual vs predicted):
#
#       TP = True Positive  = model said YES, actual was YES
#       TN = True Negative  = model said NO,  actual was NO
#       FP = False Positive = model said YES, actual was NO   (false alarm)
#       FN = False Negative = model said NO,  actual was YES  (missed)
#
#   STEP 2 — Precision (of those I said YES, how many were right?):
#
#       Precision = TP / (TP + FP)
#
#   STEP 3 — Recall (of those actually YES, how many did I catch?):
#
#       Recall = TP / (TP + FN)
#
#   STEP 4 — F1-SCORE (harmonic mean of Precision and Recall):
#
#       F1 = 2 · (Precision · Recall) / (Precision + Recall)
#
#   Equivalent single-line form (algebraically identical):
#
#       F1 = 2·TP / (2·TP + FP + FN)
#
# SYMBOLS:
#   TP, TN, FP, FN  = counts from confusion matrix
#   Precision       = correctness of YES predictions
#   Recall          = coverage of actual YES cases
#   F1              = balanced score (0 = terrible, 1 = perfect)
#   ·               = multiply
#
# ============================================================================
# WHY IT MATTERS — ACCURACY vs F1
# ============================================================================
#
#   Scenario: 100 applicants. 95 won't register, 5 will.
#
#   Model behavior              Accuracy   Precision  Recall   F1
#   --------------------------- --------- ---------- -------- -----
#   Always say NO (lazy)          95.0%       —         0.0    0.0
#   Always say YES (reckless)     5.0%      0.05       1.0    0.095
#   Balanced good model           92.0%      0.83       0.83   0.83
#
#   ACCURACY LIES. F1 TELLS THE TRUTH.
#
# ============================================================================
# MANUAL CALCULATION — YOUR JIGJIGA REGISTRAR DATA
# ============================================================================
#
#   DATA (10 applicants, from last term):
#     Actual: whether the student ACTUALLY registered (1=Yes, 0=No)
#     Predicted: what the classifier said BEFORE registration
#
#     #   Actual   Predicted
#     --  ------   ---------
#      1     1         1     →  TP
#      2     1         1     →  TP
#      3     1         1     →  TP
#      4     1         1     →  TP
#      5     1         1     →  TP
#      6     1         0     →  FN  (missed!)
#      7     1         0     →  FN  (missed!)
#      8     1         0     →  FN  (missed!)
#      9     0         1     →  FP  (false alarm!)
#     10     0         0     →  TN
#
#   STEP 1 — Count from confusion matrix:
#     TP = 5   (positions 1-5)
#     FN = 3   (positions 6-8)
#     FP = 1   (position 9)
#     TN = 1   (position 10)
#
#   STEP 2 — Precision = TP / (TP + FP) = 5 / (5 + 1) = 5/6 = 0.833
#
#   STEP 3 — Recall    = TP / (TP + FN) = 5 / (5 + 3) = 5/8 = 0.625
#
#   STEP 4 — F1 = 2 · (P · R) / (P + R)
#              = 2 · (0.833 · 0.625) / (0.833 + 0.625)
#              = 2 · 0.5208 / 1.4583
#              = 1.0417 / 1.4583
#              = 0.714
#
#   SAME via short form:
#     F1 = 2·TP / (2·TP + FP + FN)
#        = 10 / (10 + 1 + 3)
#        = 10 / 14
#        = 0.714   ✅ matches
#
#   INTERPRETATION:
#     Precision 0.833 → "of those we predicted YES, 83% were correct"
#     Recall    0.625 → "we only caught 62% of actual YES students"
#     F1        0.714 → "balanced score — decent but room to improve"
#
#   WHY F1 < Accuracy for this model?
#     Accuracy = (TP+TN)/n = (5+1)/10 = 60%. Even lower than F1 here.
#     But in imbalanced cases F1 and Accuracy disagree — and F1 is
#     the honest one.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the actual and predicted lists.
#   NEVER edit the metric calculations or sklearn calls.
#   Both lists MUST have the same length and contain ONLY 0 or 1.
# ============================================================================


# ============================================================================
# PART 1: MANUAL VERSION — from scratch (transparent math)
# ============================================================================
#
#   No external library needed for manual version — pure Python.
#   Optional: pip install numpy (only for library version below)
#
# ============================================================================

print("=" * 68)
print("FORMULA 06: F1-SCORE  (EVALUATION METRIC)")
print("=" * 68)

# ---------- STEP 1: YOUR DATA (ONLY EDIT THIS SECTION) ----------
# actual    = ground truth (1 = registered, 0 = did not register)
# predicted = model's Yes/No decision

actual    = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]   # 8 true YES, 2 true NO
predicted = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]   # 5 TP, 3 FN, 1 FP, 1 TN

# ---------- STEP 2: CONFUSION MATRIX (NEVER EDIT) ----------
# Loop through every pair and count TP, TN, FP, FN

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(actual)):
    if actual[i] == 1 and predicted[i] == 1:
        TP = TP + 1
    elif actual[i] == 0 and predicted[i] == 0:
        TN = TN + 1
    elif actual[i] == 0 and predicted[i] == 1:
        FP = FP + 1
    elif actual[i] == 1 and predicted[i] == 0:
        FN = FN + 1

print("\n--- MANUAL VERSION ---")
print("\nSTEP 2: Confusion Matrix counts")
print(f"  TP (said YES, was YES) = {TP}")
print(f"  TN (said NO,  was NO)  = {TN}")
print(f"  FP (said YES, was NO)  = {FP}   ← false alarm")
print(f"  FN (said NO,  was YES) = {FN}   ← missed")

# ---------- STEP 3: PRECISION (NEVER EDIT) ----------
# Math:  Precision = TP / (TP + FP)

if (TP + FP) == 0:
    precision = 0.0
else:
    precision = TP / (TP + FP)

print(f"\nSTEP 3: Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.4f}")

# ---------- STEP 4: RECALL (NEVER EDIT) ----------
# Math:  Recall = TP / (TP + FN)

if (TP + FN) == 0:
    recall = 0.0
else:
    recall = TP / (TP + FN)

print(f"STEP 4: Recall    = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.4f}")

# ---------- STEP 5: F1-SCORE (NEVER EDIT) ----------
# Math:  F1 = 2·(Precision·Recall) / (Precision + Recall)

if (precision + recall) == 0:
    f1 = 0.0
else:
    f1 = 2 * (precision * recall) / (precision + recall)

print(f"STEP 5: F1 = 2·(P·R) / (P+R) = "
      f"2·({precision:.4f}·{recall:.4f}) / ({precision:.4f}+{recall:.4f}) = {f1:.4f}")

# Short form (verification)
f1_short = (2 * TP) / (2 * TP + FP + FN)
print(f"       Short form: 2·TP / (2·TP + FP + FN) = "
      f"{2*TP} / ({2*TP} + {FP} + {FN}) = {f1_short:.4f}")


# ============================================================================
# PART 2: LIBRARY VERSION — scikit-learn (production code)
# ============================================================================
#
#   Install once:  pip install scikit-learn
#   Same math inside. One line.
#
# ============================================================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

print("\n" + "=" * 68)
print("LIBRARY VERSION (scikit-learn)")
print("=" * 68)

# ---------- CONFUSION MATRIX (NEVER EDIT) ----------
# cm[i][j] = count of actual=i predicted=j
#   cm[0][0] = TN    cm[0][1] = FP
#   cm[1][0] = FN    cm[1][1] = TP

cm = confusion_matrix(actual, predicted)
print("\nConfusion Matrix:")
print(f"  TN={cm[0][0]}   FP={cm[0][1]}")
print(f"  FN={cm[1][0]}   TP={cm[1][1]}")

# ---------- METRICS (NEVER EDIT) ----------

lib_precision = precision_score(actual, predicted, zero_division=0)
lib_recall    = recall_score(actual, predicted, zero_division=0)
lib_f1        = f1_score(actual, predicted, zero_division=0)
lib_accuracy  = accuracy_score(actual, predicted)

print(f"\nPrecision = {lib_precision:.4f}")
print(f"Recall    = {lib_recall:.4f}")
print(f"F1        = {lib_f1:.4f}")
print(f"Accuracy  = {lib_accuracy:.4f}   ← lower than F1 in this case")


# ============================================================================
# PART 3: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Manual  : Precision={precision:.3f}  "
      f"Recall={recall:.3f}  F1={f1:.3f}")
print(f"  Library : Precision={lib_precision:.3f}  "
      f"Recall={lib_recall:.3f}  F1={lib_f1:.3f}")
print()
print("  WHAT THIS MEANS:")
print(f"  → Precision {precision:.2f}: of {TP+FP} students we predicted YES, "
      f"{TP} really enrolled.")
print(f"  → Recall    {recall:.2f}: of {TP+FN} students who really enrolled, "
      f"we caught {TP}.")
print(f"  → F1        {f1:.2f}: BOTH scores together. Need BOTH high to score high.")
print()
print("  DECISION FOR THE REGISTRAR:")
print("  → Precision is decent (0.83) — few false alarms")
print("  → Recall is weak (0.63) — 3 students who WOULD have registered were missed")
print("  → ACTION: Call the applicants we marked NO. Before the deadline.")
print("  → Each missed student is a lost seat — F1 just showed us where.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - the confusion matrix loop
#      - the precision / recall / F1 formulas
#      - sklearn metric calls
#
#   🔁 ALWAYS edit:
#      - actual and predicted lists
#      - Both lists MUST be SAME length
#      - Both lists MUST contain ONLY 0 or 1
#
#   ⚙️ AUTO-computed:
#      - TP, TN, FP, FN
#      - Precision, Recall, F1
#
# ============================================================================
# WHEN TO USE F1 (and when NOT to)
# ============================================================================
#
#   USE F1 WHEN:
#     → Classes are IMBALANCED (e.g., 95% No, 5% Yes)
#     → Both false alarms AND missed cases are costly
#     → You want ONE number that cannot be gamed
#
#   DO NOT USE F1 WHEN:
#     → Accuracy is fine for balanced data (50/50 classes)
#     → You only care about ONE type of error
#       (then use Precision alone or Recall alone)
#
# ============================================================================
# WUCHALE — waits for Formula 08 (K-Means, unsupervised)
# ============================================================================
