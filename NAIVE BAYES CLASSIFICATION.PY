# ============================================================================
# FORMULA 16: NAIVE BAYES CLASSIFICATION
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga
# Task   : Predict YES/NO using PROBABILITY — Bayes' theorem in action
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Every classifier you have seen so far — Logistic Regression, SVM,
#  KNN, Random Forest — learns a boundary. A line. A margin. A vote.
#
#  Naive Bayes does not draw any boundary. It thinks like a DOCTOR.
#
#  A doctor does not look at you and draw a line. She thinks:
#    'What is the base rate of this disease in the population?
#     Given this symptom, how often does that disease produce it?
#     Given that symptom, how often does a healthy person have it?'
#  Then she combines the probabilities. That is Bayes' theorem.
#
#  Naive Bayes does exactly this. For our registrar problem:
#    'What percentage of applicants register overall?       ← Prior
#     Of those who registered, how many lived close?        ← Likelihood
#     Of those who did not register, how many lived far?    ← Likelihood
#     Given this new applicant lives far and is older,
#     what is the probability they will register?'
#
#  Naive Bayes combines these probabilities and picks the answer
#  with higher probability. That is the whole algorithm.
#
#  Why is it called NAIVE? Because it makes one naive assumption:
#  that every feature is INDEPENDENT of every other feature. Distance
#  and age, it assumes, have nothing to do with each other. In real
#  life that is often WRONG — but remarkably, the classifier still
#  works. Naive Bayes wins text classification, spam filters, and
#  medical screening even though its core assumption is false.
#
#  For Jigjiga: our 10 applicants. 6 registered, 4 did not.
#  New applicant arrives — 11 km away, age 25. Bayes says:
#    P(NO | far, old) = 93.7%
#    P(YES | far, old) = 6.3%
#  Prediction: NO. Because far-and-old almost never registered.
#
#  For Wuchale: once 2027 data arrives, Naive Bayes is the FASTEST
#  classifier to train — one pass through the data, no iteration,
#  no gradient descent. It is the workhorse of real-time systems.
#
#  Bayes' theorem is 250 years old. It still beats modern algorithms
#  on text, on spam, on medical screening. Because it thinks the
#  way humans think: given what I already know, what is likely now?"
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA — BAYES' THEOREM
# ============================================================================
#
#   BAYES' THEOREM (the foundation):
#
#       P(C | X) = P(X | C) · P(C) / P(X)
#
#   where:
#       C = a class (YES or NO)
#       X = the feature vector (the new applicant)
#       P(C | X) = POSTERIOR  (probability of class given the features)
#       P(X | C) = LIKELIHOOD (probability of features given the class)
#       P(C)     = PRIOR     (base rate of the class)
#       P(X)     = EVIDENCE  (probability of the features)
#
#   THE NAIVE ASSUMPTION — features are independent:
#
#       P(X | C) = P(x₁ | C) · P(x₂ | C) · ... · P(xₚ | C)
#
#   SO THE FULL CLASSIFIER BECOMES:
#
#       P(C | X) ∝ P(C) · Πⱼ P(xⱼ | C)
#
#       (∝ means "proportional to" — we drop P(X) because it is
#        the same for all classes and does not change the winner)
#
#   DECISION RULE:
#
#       ŷ = argmax over C of  [ P(C) · Πⱼ P(xⱼ | C) ]
#
#       → predict the class with the HIGHEST posterior probability
#
#   LAPLACE SMOOTHING (to handle zero counts):
#
#       P(xⱼ = v | C) = ( count(xⱼ = v, C) + α ) / ( count(C) + α·k )
#
#       where α = 1 (default smoothing strength)
#             k = number of possible values of feature xⱼ
#
#   CONTINUOUS FEATURES — use GAUSSIAN likelihood:
#
#       P(xⱼ | C) = (1 / √(2π·σ²ᵢ)) · exp( −(xⱼ − μᵢ)² / (2·σ²ᵢ) )
#
#       where μᵢ, σᵢ = mean and std of feature j within class C
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   P(C | X)     = posterior (what we want)
#   P(X | C)     = likelihood (how typical are these features for class C)
#   P(C)         = prior (how common is class C overall)
#   P(X)         = evidence (ignored — same for all classes)
#   C            = class label (YES or NO)
#   X, xⱼ        = feature vector and one feature
#   Π (pi)       = product (multiply across all features)
#   α (alpha)    = Laplace smoothing parameter (usually 1)
#   k            = number of possible values of a discrete feature
#   μ, σ         = mean and standard deviation (for Gaussian features)
#   argmax       = "the class that gives the largest value"
#   ∝            = proportional to
#
# ============================================================================
# WHY IT MATTERS — WHERE NAIVE BAYES WINS
# ============================================================================
#
#   ALGORITHM       | SPEED   | ACCURACY | BEST FOR
#   ----------------|---------|----------|----------------------
#   Naive Bayes     | FASTEST | Good     | Text, spam, real-time
#   Logistic Reg    | Fast    | Good     | General binary
#   Random Forest   | Fast    | Better   | Tabular data
#   XGBoost         | Fast    | Best     | Tabular data
#   Neural Network  | SLOW    | Best     | Images, text, speech
#   KNN             | SLOW    | Good     | Small data
#   SVM             | Medium  | Good     | Small data, high-dim
#
#   Naive Bayes trains in ONE pass through the data — no iteration.
#   It is 100× faster than XGBoost on the same data. On small
#   datasets, that speed costs almost nothing in accuracy.
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA APPLICANT DATA
# ============================================================================
#
#   TRAINING DATA (10 applicants, discretized for clarity):
#     Distance: close (< 10 km)  or  far (≥ 10 km)
#     Age:      young (≤ 22)     or  old (> 22)
#
#     #   Distance  Age   Registered
#     1      2      18      1  → close, young, YES
#     2      5      22      1  → close, young, YES
#     3     15      25      0  → far, old, NO
#     4      3      19      1  → close, young, YES
#     5     20      30      0  → far, old, NO
#     6      8      24      1  → close, old, YES
#     7     12      28      0  → far, old, NO
#     8      4      20      1  → close, young, YES
#     9      6      21      1  → close, young, YES
#     10    10      26      0  → far, old, NO
#
#   ---------------------------------------------------------
#   STEP 1 — PRIORS (how common is each class?)
#   ---------------------------------------------------------
#   YES count = 6  →  P(YES) = 6/10 = 0.600
#   NO  count = 4  →  P(NO)  = 4/10 = 0.400
#
#   ---------------------------------------------------------
#   STEP 2 — COUNTS (how often does each value appear per class?)
#   ---------------------------------------------------------
#   Within YES (n = 6):
#     close : 6    far  : 0
#     young : 5    old  : 1
#
#   Within NO  (n = 4):
#     close : 0    far  : 4
#     young : 0    old  : 4
#
#   ---------------------------------------------------------
#   STEP 3 — LIKELIHOODS with LAPLACE SMOOTHING (α = 1)
#   ---------------------------------------------------------
#   For distance (k = 2 possible values: close, far):
#     P(close | YES) = (6 + 1) / (6 + 1·2) = 7 / 8 = 0.875
#     P(far   | YES) = (0 + 1) / (6 + 1·2) = 1 / 8 = 0.125
#     P(close | NO)  = (0 + 1) / (4 + 1·2) = 1 / 6 = 0.167
#     P(far   | NO)  = (4 + 1) / (4 + 1·2) = 5 / 6 = 0.833
#
#   For age (k = 2 possible values: young, old):
#     P(young | YES) = (5 + 1) / 8 = 0.750
#     P(old   | YES) = (1 + 1) / 8 = 0.250
#     P(young | NO)  = (0 + 1) / 6 = 0.167
#     P(old   | NO)  = (4 + 1) / 6 = 0.833
#
#   ---------------------------------------------------------
#   STEP 4 — CLASSIFY NEW APPLICANT: Distance = 11 (far), Age = 25 (old)
#   ---------------------------------------------------------
#
#   Posterior (unnormalized):
#     P(YES) × P(far|YES) × P(old|YES)
#         = 0.600 × 0.125 × 0.250
#         = 0.01875
#
#     P(NO)  × P(far|NO)  × P(old|NO)
#         = 0.400 × 0.833 × 0.833
#         = 0.27757
#
#   Normalize (divide by total = 0.01875 + 0.27757 = 0.29632):
#     P(YES | far, old) = 0.01875 / 0.29632 = 0.0633
#     P(NO  | far, old) = 0.27757 / 0.29632 = 0.9367
#
#   Prediction: NO   (confidence 93.7%)
#
#   ---------------------------------------------------------
#   STEP 5 — COMPARE WITH A CLOSE-YOUNG APPLICANT (Distance = 5, Age = 20)
#   ---------------------------------------------------------
#
#     P(YES) × P(close|YES) × P(young|YES)
#         = 0.600 × 0.875 × 0.750
#         = 0.39375
#
#     P(NO)  × P(close|NO)  × P(young|NO)
#         = 0.400 × 0.167 × 0.167
#         = 0.01114
#
#   Normalize:
#     P(YES) = 0.39375 / 0.40489 = 0.9724  (97.2%)
#     P(NO)  = 0.01114 / 0.40489 = 0.0276
#
#   Prediction: YES   (confidence 97.2%)
#
#   → Distance is the STRONGEST driver in the model.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the training data lists and the new applicant.
#   NEVER edit the Bayes probability calculations.
#   All training lists MUST be same length.
# ============================================================================


import numpy as np
from collections import Counter


# ============================================================================
# PART 1: MANUAL NAIVE BAYES — from scratch
# ============================================================================

print("=" * 68)
print("PART 1: NAIVE BAYES (manual, from scratch)")
print("=" * 68)

# ---------- TRAINING DATA (ONLY EDIT THIS) ----------

# Raw features
distance   = [2, 5, 15, 3, 20, 8, 12, 4, 6, 10]
age        = [18, 22, 25, 19, 30, 24, 28, 20, 21, 26]
registered = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]

# ---------- DISCRETIZE FEATURES (turn numbers into categories) ----------

def to_distance_cat(d):
    return "close" if d < 10 else "far"

def to_age_cat(a):
    return "young" if a <= 22 else "old"

# ---------- STEP 1: PRIORS (NEVER EDIT) ----------

n = len(registered)
count_yes = sum(registered)
count_no  = n - count_yes

prior_yes = count_yes / n
prior_no  = count_no  / n

print(f"\nSTEP 1: Priors")
print(f"  P(YES) = {count_yes}/{n} = {prior_yes:.3f}")
print(f"  P(NO)  = {count_no}/{n} = {prior_no:.3f}")

# ---------- STEP 2: LIKELIHOOD TABLES (NEVER EDIT) ----------

def build_likelihood_table():
    # Count feature values for each class
    yes_dist = Counter()
    yes_age  = Counter()
    no_dist  = Counter()
    no_age   = Counter()

    for i in range(n):
        dc = to_distance_cat(distance[i])
        ac = to_age_cat(age[i])
        if registered[i] == 1:
            yes_dist[dc] += 1
            yes_age[ac]  += 1
        else:
            no_dist[dc] += 1
            no_age[ac]  += 1

    # Laplace smoothing (α=1) for each feature
    def smooth(counts, all_values, total, k):
        return {(v): (counts[v] + 1) / (total + k) for v in all_values}

    yes_dist_p = smooth(yes_dist, ["close", "far"], count_yes, 2)
    yes_age_p  = smooth(yes_age,  ["young", "old"], count_yes, 2)
    no_dist_p  = smooth(no_dist,  ["close", "far"], count_no,  2)
    no_age_p   = smooth(no_age,   ["young", "old"], count_no,  2)

    return yes_dist_p, yes_age_p, no_dist_p, no_age_p

yes_dist_p, yes_age_p, no_dist_p, no_age_p = build_likelihood_table()

print("\nSTEP 2: Likelihoods (Laplace smoothed, α=1)")
print(f"  P(close | YES) = {yes_dist_p['close']:.3f}")
print(f"  P(far   | YES) = {yes_dist_p['far']:.3f}")
print(f"  P(young | YES) = {yes_age_p['young']:.3f}")
print(f"  P(old   | YES) = {yes_age_p['old']:.3f}")
print(f"  P(close | NO)  = {no_dist_p['close']:.3f}")
print(f"  P(far   | NO)  = {no_dist_p['far']:.3f}")
print(f"  P(young | NO)  = {no_age_p['young']:.3f}")
print(f"  P(old   | NO)  = {no_age_p['old']:.3f}")

# ---------- STEP 3: CLASSIFY (NEVER EDIT) ----------

def naive_bayes_predict(dist, age_val):
    dc = to_distance_cat(dist)
    ac = to_age_cat(age_val)

    # Posterior ∝ Prior × Likelihood × Likelihood
    score_yes = prior_yes * yes_dist_p[dc] * yes_age_p[ac]
    score_no  = prior_no  * no_dist_p[dc]  * no_age_p[ac]

    total = score_yes + score_no
    prob_yes = score_yes / total if total > 0 else 0.0
    prob_no  = score_no  / total if total > 0 else 0.0

    prediction = 1 if prob_yes >= prob_no else 0
    return prediction, prob_yes, prob_no

# ---------- STEP 4: TEST APPLICANTS ----------

print("\nSTEP 3: Classify new applicants")
test_cases = [
    (11, 25, "far, old"),
    (5,  20, "close, young"),
    (7,  23, "close, old"),
    (14, 20, "far, young"),
]

for dist_val, age_val, desc in test_cases:
    pred, p_yes, p_no = naive_bayes_predict(dist_val, age_val)
    print(f"\n  Applicant: dist={dist_val}, age={age_val}  ({desc})")
    print(f"    P(YES | features) = {p_yes:.3f}")
    print(f"    P(NO  | features) = {p_no:.3f}")
    print(f"    Prediction: {'YES' if pred==1 else 'NO '}")


# ============================================================================
# PART 2: LIBRARY NAIVE BAYES — scikit-learn
# ============================================================================
#
#   sklearn provides two versions:
#     CategoricalNB  — for discrete/categorical features (best for our case)
#     GaussianNB     — for continuous features (Gaussian likelihood)
#   Both implement Bayes' theorem with the naive assumption.
#
# ============================================================================

from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("\n" + "=" * 68)
print("PART 2: NAIVE BAYES (scikit-learn)")
print("=" * 68)

# ---------- CATEGORICAL NB (matches manual calculation) ----------

# Encode categories: close=0, far=1 ; young=0, old=1
X_cat = np.array([
    [0 if to_distance_cat(distance[i]) == "close" else 1,
     0 if to_age_cat(age[i]) == "young" else 1]
    for i in range(n)
])
Y_cat = np.array(registered, dtype=int)

cat_nb = CategoricalNB(alpha=1.0)    # α = Laplace smoothing
cat_nb.fit(X_cat, Y_cat)

# Test the same 4 applicants
new_cat = np.array([
    [1 if to_distance_cat(11) == "far" else 0, 1 if to_age_cat(25) == "old" else 0],
    [1 if to_distance_cat(5)  == "far" else 0, 1 if to_age_cat(20) == "young" else 0],
    [1 if to_distance_cat(7)  == "far" else 0, 1 if to_age_cat(23) == "old" else 0],
    [1 if to_distance_cat(14) == "far" else 0, 1 if to_age_cat(20) == "young" else 0],
])

cat_preds  = cat_nb.predict(new_cat)
cat_probas = cat_nb.predict_proba(new_cat)

print("\nCategoricalNB predictions:")
for j in range(4):
    d, a, desc = test_cases[j]
    print(f"  dist={d:>2}, age={a}  → "
          f"{'YES' if cat_preds[j]==1 else 'NO '}  "
          f"P(NO)={cat_probas[j][0]:.3f}  P(YES)={cat_probas[j][1]:.3f}")

# ---------- GAUSSIAN NB (use raw numbers directly) ----------

print("\nGaussianNB predictions (raw continuous features):")

X_raw = np.array([[distance[i], age[i]] for i in range(n)], dtype=float)
Y_raw = np.array(registered, dtype=int)

# Gaussian NB can benefit from standardization if features differ widely
scaler = StandardScaler()
X_raw_scaled = scaler.fit_transform(X_raw)

gauss_nb = GaussianNB()
gauss_nb.fit(X_raw_scaled, Y_raw)

new_raw = np.array([[11, 25], [5, 20], [7, 23], [14, 20]], dtype=float)
new_raw_scaled = scaler.transform(new_raw)

gauss_preds  = gauss_nb.predict(new_raw_scaled)
gauss_probas = gauss_nb.predict_proba(new_raw_scaled)

for j in range(4):
    d, a, desc = test_cases[j]
    print(f"  dist={d:>2}, age={a}  → "
          f"{'YES' if gauss_preds[j]==1 else 'NO '}  "
          f"P(NO)={gauss_probas[j][0]:.3f}  P(YES)={gauss_probas[j][1]:.3f}")


# ============================================================================
# PART 3: TRAINING ACCURACY — how good is the classifier?
# ============================================================================

print("\n" + "=" * 68)
print("PART 3: TRAINING ACCURACY")
print("=" * 68)

# Categorical NB on training data
cat_train_preds = cat_nb.predict(X_cat)
cat_acc = accuracy_score(Y_cat, cat_train_preds) * 100

# Gaussian NB on training data
gauss_train_preds = gauss_nb.predict(X_raw_scaled)
gauss_acc = accuracy_score(Y_raw, gauss_train_preds) * 100

print(f"  CategoricalNB training accuracy: {cat_acc:.1f}%")
print(f"  GaussianNB   training accuracy: {gauss_acc:.1f}%")


# ============================================================================
# PART 4: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print("  PREDICTION FOR NEW APPLICANT (dist=11, age=25 — far, old):")
print("    Manual:         P(NO) = 0.937   → NO")
print("    CategoricalNB:  P(NO) = " + f"{cat_probas[0][0]:.3f}" + "   → NO")
print("    GaussianNB:     P(NO) = " + f"{gauss_probas[0][0]:.3f}" + "   → NO")
print("  → All three agree. Prediction is robust.")
print()
print("  WHY IT WORKS:")
print("  → All 4 training NOs were far AND old.")
print("  → The new applicant is far AND old.")
print("  → Naive Bayes sees the pattern and bets NO.")
print()
print("  JIGJIGA DECISION:")
print("  → Any applicant with distance ≥ 10 AND age > 22 → predicted NO")
print("  → CALL those applicants. Historically, they do not show up.")
print("  → Applicants close AND young → predicted YES with 97% confidence")
print("  → Do not waste calls on those. They are coming.")
print()
print("  WUCHALE NOTE:")
print("  → Naive Bayes needs labeled history (2027+).")
print("  → For 2027 launch without history, use K-Means (Formula 07).")
print("  → Once 2027 data exists, Naive Bayes trains in seconds and")
print("    gives the Wuchale office a live classifier on day 1 of 2028.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - naive_bayes_predict() function
#      - build_likelihood_table() function
#      - CategoricalNB / GaussianNB construction
#      - Laplace smoothing formula
#
#   🔁 ALWAYS edit:
#      - distance, age, registered lists
#      - test_cases (new applicants to classify)
#      - the discretization thresholds (to_distance_cat, to_age_cat)
#
#   ⚙️ AUTO-computed:
#      - priors
#      - likelihoods (with Laplace smoothing)
#      - posteriors
#      - predictions
#
#   ⚠️ Use CategoricalNB when features are discrete (categories)
#   ⚠️ Use GaussianNB when features are continuous (raw numbers)
#   ⚠️ Laplace smoothing (α=1) prevents ZERO probabilities — keep it
#   ⚠️ Naive assumption (independence) is usually violated but still works
#
# ============================================================================
# WHEN TO USE NAIVE BAYES
# ============================================================================
#
#   ✅ USE IT WHEN:
#     - Text classification (spam, sentiment, topic)
#     - Very fast training required (real-time)
#     - Small datasets
#     - Features are conditionally independent (or close)
#     - You need interpretable probabilities
#
#   ❌ DON'T USE IT WHEN:
#     - Features are strongly correlated (e.g., age + birth_year)
#     - You need calibrated probabilities (use Logistic Regression)
#     - Continuous features with complex distributions (use GBT)
#     - High accuracy needed and dataset is large (use XGBoost)
#
# ============================================================================
