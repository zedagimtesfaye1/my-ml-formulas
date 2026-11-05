# ============================================================================
# FORMULA 05: LOGISTIC REGRESSION  (FIRST CLASSIFICATION FORMULA)
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga, Ethiopia
# Task   : Predict YES / NO (not a number — a decision)
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Until now, every formula answered: HOW MANY? 56 students. 48 students.
#  That is REGRESSION.
#
#  Today we change the question. Not HOW MANY — but YES or NO.
#
#  Will this applicant complete registration?  YES or NO.
#  Will this student attend next term?         YES or NO.
#  Will this document pass verification?       YES or NO.
#
#  Logistic Regression is the FIRST machine learning formula that
#  gives you a DECISION, not a number.
#
#  Here is the magic. Instead of predicting a raw number, it predicts
#  a PROBABILITY — a number between 0 and 1.
#    0.0  = definitely NO
#    1.0  = definitely YES
#    0.5  = we do not know — coin flip
#
#  And then: if probability >= 0.5, we say YES. Else we say NO.
#
#  For Jigjiga: we ask Logistic Regression — 'will this student
#  complete registration or not, based on their age and distance
#  from school?'  The model learns: YOUNGER = more likely to enroll.
#  FAR AWAY = less likely. And it gives each applicant a % chance.
#
#  Now you do not just count students. You know WHICH ones will come,
#  and WHICH ones you need to call. That is the power of classification."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA
# ============================================================================
#
#   STEP 1 (linear part — same as multiple regression):
#       z  =  W₁·X₁ + W₂·X₂ + ... + Wₚ·Xₚ + B
#
#   STEP 2 (sigmoid — squashes z into 0..1):
#       ŷ  =  1 / (1 + e^(−z))
#
#   STEP 3 (decision — turn probability into YES/NO):
#       if ŷ >= 0.5  →  YES (class 1)
#       if ŷ <  0.5  →  NO  (class 0)
#
#   LEARNING (gradient descent on binary cross-entropy):
#       Wᵢ_new = Wᵢ_old − α · (1/n) · Σ[ (ŷ − Y) · Xᵢ ]
#       B_new  = B_old  − α · (1/n) · Σ[  ŷ − Y       ]
#
# SYMBOLS:
#   z  = linear score (real number, any range)
#   ŷ  = probability (0 to 1)
#   e  = Euler's number ≈ 2.718 (constant)
#   W₁…Wₚ = weights learned from data
#   B  = bias (intercept)
#   X₁…Xₚ = features
#   Y  = true label (0 or 1)
#   n  = number of data points
#   α  = learning rate
#   Σ  = sum over all data points
#
# ============================================================================
# WHY IT MATTERS — REGRESSION vs CLASSIFICATION
# ============================================================================
#
#   QUESTION                       | FORMULA              | OUTPUT
#   -------------------------------|----------------------|--------------
#   HOW MANY students in 2027?    | Linear Regression    | 56 (a number)
#   WILL THIS student enroll?     | Logistic Regression  | YES/NO (+ %)
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA REGISTRAR DATA
# ============================================================================
#
#   NEW DATASET (registrar office — 8 applicants in 2026):
#   Each row = one applicant
#
#     X₁ = Distance from home to school (km)
#     X₂ = Age (years)
#     Y  = Completed registration?  (1 = Yes, 0 = No)
#
#     #   Distance  Age    Completed?
#     --  --------  ----   ----------
#     1      2       18        1  (Yes)
#     2      5       22        1
#     3     15       25        0  (No)
#     4      3       19        1
#     5     20       30        0
#     6      8       24        1
#     7     12       28        0
#     8      4       20        1
#
#   Start: W₁=0, W₂=0, B=0, α=0.01
#
#   ONE STEP (applicant #1: X₁=2, X₂=18, Y=1):
#     z  = 0·2 + 0·18 + 0 = 0
#     ŷ  = 1/(1+e^0) = 1/(1+1) = 0.50
#     error = ŷ − Y = 0.50 − 1 = −0.50
#
#     gradient_W₁ = error · X₁ = −0.50 · 2  = −1.00
#     gradient_W₂ = error · X₂ = −0.50 · 18 = −9.00
#     gradient_B  = error       = −0.50
#
#     W₁ = 0 − 0.01·(−1.00) = 0.010
#     W₂ = 0 − 0.01·(−9.00) = 0.090
#     B  = 0 − 0.01·(−0.50) = 0.005
#
#   ...repeat over all 8 applicants, ~1000 times total...
#
#   FINAL LEARNED MODEL (approximately, after full training):
#     W₁ ≈ −0.35   (farther distance → lower probability)
#     W₂ ≈ +0.20   (older age here → slightly lower in our data,
#                   but model will find the true pattern)
#     B  ≈ −1.0
#
#   The KEY INSIGHT: W₁ is NEGATIVE.
#   "Every 1 km farther, probability of completing drops."
#   W₂ SIGN shows the age effect direction. We let the data speak.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the DATA lists below (distance, age, registered).
#   NEVER edit the sigmoid, gradient descent, or sklearn calls.
#   All 3 lists MUST have the same length.
# ============================================================================


# ============================================================================
# PART 1: MANUAL VERSION — from scratch (transparent math)
# ============================================================================
#
#   Install once:  pip install numpy
#
# ============================================================================

import numpy as np

print("=" * 68)
print("FORMULA 05: LOGISTIC REGRESSION")
print("=" * 68)

# ---------- STEP 1: YOUR DATA (ONLY EDIT THIS SECTION) ----------
# Each row = one applicant. Y = 1 (Yes) or 0 (No).

distance   = [2,  5,  15, 3,  20, 8,  12, 4]      # X₁
age        = [18, 22, 25, 19, 30, 24, 28, 20]     # X₂
registered = [1,  1,  0,  1,  0,  1,  0,  1]      # Y  (true label)

alpha       = 0.01
iterations  = 2000

# ---------- STEP 2: BUILD FEATURE MATRIX (NEVER EDIT) ----------
# X = [[distance, age], ...]   shape (n, 2)

X = []
for i in range(len(distance)):
    X.append([distance[i], age[i]])
X = np.array(X, dtype=float)

Y = np.array(registered, dtype=float)
n = len(Y)

# ---------- STEP 3: SIGMOID FUNCTION (NEVER EDIT) ----------
# Math:  sigmoid(z) = 1 / (1 + e^(−z))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ---------- STEP 4: GRADIENT DESCENT TRAINING (NEVER EDIT) ----------
# Math:
#   z  = X·W + B
#   ŷ  = sigmoid(z)
#   gradient_W = (1/n) · Σ[(ŷ − Y) · X]
#   gradient_B = (1/n) · Σ[ ŷ − Y ]
#   W = W − α · gradient_W
#   B = B − α · gradient_B

W = np.zeros(X.shape[1])       # W₁, W₂ start at 0
B = 0.0

for step in range(iterations):
    z    = X @ W + B           # linear part
    yhat = sigmoid(z)          # probabilities for all rows

    error      = yhat - Y                 # (ŷ − Y)
    gradient_W = (1 / n) * (X.T @ error)  # (1/n) · Σ[(ŷ−Y)·X]
    gradient_B = (1 / n) * np.sum(error)  # (1/n) · Σ[ŷ−Y]

    W = W - alpha * gradient_W
    B = B - alpha * gradient_B

print("\n--- MANUAL VERSION (numpy from scratch) ---")
print(f"\nLearned W₁ (distance) = {W[0]:+.4f}")
print(f"Learned W₂ (age)      = {W[1]:+.4f}")
print(f"Learned B             = {B:+.4f}")

# ---------- STEP 5: MAKE PREDICTIONS (NEVER EDIT) ----------

probabilities = sigmoid(X @ W + B)
predictions   = (probabilities >= 0.5).astype(int)

print("\nEach applicant's probability and decision:")
print("  #   Dist  Age   Actual   Probability   Decision")
for i in range(n):
    print(f"  {i+1}   {distance[i]:>3}   {age[i]:>3}      {int(Y[i])}       "
          f"{probabilities[i]:.3f}         {'YES' if predictions[i]==1 else 'NO '}")

# ---------- STEP 6: ACCURACY (NEVER EDIT) ----------

correct = np.sum(predictions == Y)
accuracy = correct / n * 100

print(f"\nCorrect predictions : {correct} / {n}")
print(f"Accuracy            : {accuracy:.1f}%")


# ============================================================================
# PART 2: LIBRARY VERSION — scikit-learn (one line, production)
# ============================================================================
#
#   Install once:  pip install scikit-learn
#   Same math inside. sklearn adds B automatically.
#
# ============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("\n" + "=" * 68)
print("LIBRARY VERSION (scikit-learn LogisticRegression)")
print("=" * 68)

# ---------- TRAIN (NEVER EDIT) ----------

model = LogisticRegression(max_iter=2000)
model.fit(X, Y)

# ---------- EXTRACT LEARNED VALUES ----------
#   model.coef_[0][0]  = W₁
#   model.coef_[0][1]  = W₂
#   model.intercept_[0]= B

print(f"\nLearned W₁ (distance) = {model.coef_[0][0]:+.4f}")
print(f"Learned W₂ (age)      = {model.coef_[0][1]:+.4f}")
print(f"Learned B             = {model.intercept_[0]:+.4f}")

# ---------- PREDICT (NEVER EDIT) ----------

lib_prob = model.predict_proba(X)[:, 1]   # probability of YES (class 1)
lib_pred = model.predict(X)               # 0 or 1

print("\nEach applicant's probability and decision:")
print("  #   Dist  Age   Actual   Probability   Decision")
for i in range(n):
    print(f"  {i+1}   {distance[i]:>3}   {age[i]:>3}      {int(Y[i])}       "
          f"{lib_prob[i]:.3f}         {'YES' if lib_pred[i]==1 else 'NO '}")

# ---------- ACCURACY + CONFUSION MATRIX ----------

print(f"\nAccuracy = {accuracy_score(Y, lib_pred) * 100:.1f}%")

cm = confusion_matrix(Y, lib_pred)
print("\nConfusion Matrix:")
print(f"  True NO   →  predicted NO: {cm[0][0]}   |  predicted YES: {cm[0][1]}")
print(f"  True YES  →  predicted NO: {cm[1][0]}   |  predicted YES: {cm[1][1]}")


# ============================================================================
# PART 3: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Manual W₁ = {W[0]:+.3f}   W₂ = {W[1]:+.3f}   B = {B:+.3f}")
print(f"  Library W₁ = {model.coef_[0][0]:+.3f}   "
      f"W₂ = {model.coef_[0][1]:+.3f}   B = {model.intercept_[0]:+.3f}")
print()
print("  INTERPRETATION:")
if W[0] < 0:
    print(f"  → Every 1 km farther from school DECREASES chance of registering")
    print(f"  → Each km costs about {abs(W[0])*100:.0f}% probability")
if W[1] < 0:
    print(f"  → Older applicants are LESS likely to register (in this data)")
else:
    print(f"  → Older applicants are MORE likely to register (in this data)")
print()
print("  DECISION RULE:")
print("  → Probability >= 0.5 : applicant likely to register  →  plan for them")
print("  → Probability <  0.5 : applicant at risk             →  follow up by phone")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - sigmoid() function
#      - the gradient descent loop
#      - sklearn LogisticRegression calls
#
#   🔁 ALWAYS edit:
#      - distance, age, registered lists
#      - All 3 lists must be SAME length
#      - registered values MUST be 0 or 1 (Yes=1, No=0)
#
#   ⚙️ AUTO-computed:
#      - W₁, W₂, B
#      - probabilities
#      - accuracy, confusion matrix
#
# ============================================================================
# WUCHALE — waits for Formula 08 (K-Means, unsupervised)
# ============================================================================
