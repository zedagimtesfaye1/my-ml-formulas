# ============================================================================
# FORMULA 11: SHANNON ENTROPY  —  H(X) = −Σ p(x) · log₂(p(x))
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga
# Task   : Measure IMPURITY / MIXEDNESS — the brain of Decision Trees
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Shannon invented this in 1948 to measure information in phone signals.
#  Seventy years later, it became the brain of decision trees. Why?
#  Because entropy is the mathematical way to ask ONE question:
#  'How MIXED are these things?'
#
#  Imagine two classrooms. In Room A, all 30 students will register —
#  every single one. In Room B, exactly half will register, half won't.
#  Which classroom is more UNCERTAIN? Room B. Every student is a coin
#  flip. Room A is not — you already know the answer.
#
#  Entropy measures that uncertainty. It is ZERO when everyone is the
#  same. It is MAXIMUM when everything is fifty-fifty. In between, it
#  tells you exactly how mixed the room is — in BITS.
#
#  For Jigjiga: our applicants are 60% registered, 40% not. Entropy
#  = 0.971 bits. Close to maximum (1.0). Meaning our applicants are
#  very MIXED. The problem is HARD. A lazy model cannot just guess
#  YES — it would be wrong 40% of the time.
#
#  Decision Trees use this. At every split, they ask: 'If I split
#  the students by distance, does entropy DROP?' The split that drops
#  entropy the most is the split we take. That drop has a name:
#  INFORMATION GAIN. It is the engine of the entire tree.
#
#  Every GPT, every classifier, every ranking algorithm — all of them
#  minimize some form of entropy. Cross-entropy. KL divergence.
#  Information gain. Different names. Same idea.
#
#  For Wuchale: how mixed is our student pool? High entropy means
#  we need MANY strategies. Low entropy means ONE strategy works.
#  Entropy tells us BEFORE we spend a single birr."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA — SHANNON ENTROPY
# ============================================================================
#
#   H(X) = − Σ p(xᵢ) · log₂( p(xᵢ) )       for i = 1 to k classes
#
#   Equivalent (multiplication is commutative):
#   H(X) = − Σ p(xᵢ) · log₂( p(xᵢ) ) = Σ p(xᵢ) · (−log₂(p(xᵢ)))
#
#   Convention: 0 · log(0) = 0   (a class with p=0 contributes nothing)
#
#   INFORMATION GAIN (how Decision Trees use entropy):
#
#   IG = H(parent)  −  Σ [ (|child| / |parent|) · H(child) ]
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   H(X)     = entropy of variable X (in bits)
#   p(xᵢ)    = probability of class i = countᵢ / total
#   k        = number of classes
#   log₂     = logarithm base 2 (result is in BITS)
#   Σ        = sum over all classes
#   −        = minus sign (because log₂ of a probability is negative)
#   IG       = Information Gain (entropy drop from a split)
#   |child|  = number of samples in a child group
#   |parent| = number of samples total
#
# ============================================================================
# WHY IT MATTERS — IT POWERS DECISION TREES AND DEEP LEARNING
# ============================================================================
#
#   FORMULA USED IN:
#     - Decision Trees      (split by Information Gain = entropy drop)
#     - Random Forests      (many trees, all using entropy)
#     - Gradient Boosting   (XGBoost, LightGBM)
#     - Cross-entropy loss  (every classifier, every neural network)
#     - Feature selection   (high entropy = weak feature)
#
#   ENTROPY TELLS YOU:
#     H = 0.0   → perfect purity (all same class) → split not needed
#     H = 1.0   → binary maximum (50/50)          → hardest case
#     H > 1.0   → multi-class case
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA DATA (5 CASES)
# ============================================================================
#
#   CASE 1 — 6 enrolled, 4 not (from Formula 05 data)
#   ---------------------------------------------
#   Total = 10
#   p(yes) = 6/10 = 0.6 , p(no) = 4/10 = 0.4
#
#   log₂(0.6) = -0.7370
#   log₂(0.4) = -1.3219
#
#   H = − (0.6)(−0.7370) − (0.4)(−1.3219)
#     =  0.4422 + 0.5288
#     =  0.9710 bits
#
#   → Near maximum (1.0). Very mixed. Hard problem.
#
#   CASE 2 — 5 enrolled, 5 not (uniform)
#   ---------------------------------------------
#   p = 0.5, 0.5
#   log₂(0.5) = -1
#   H = − 0.5·(−1) − 0.5·(−1) = 0.5 + 0.5 = 1.0 bit   ← MAXIMUM
#
#   CASE 3 — 10 enrolled, 0 not (pure)
#   ---------------------------------------------
#   p = 1.0, 0.0
#   H = −1·log₂(1) − 0 = 0 bits   ← MINIMUM
#
#   CASE 4 — Jigjiga 2022 gender (50 male, 30 female)
#   ---------------------------------------------
#   p(M) = 50/80 = 0.625 , p(F) = 30/80 = 0.375
#
#   log₂(0.625) = -0.6781
#   log₂(0.375) = -1.4150
#
#   H = −(0.625)(−0.6781) − (0.375)(−1.4150)
#     =  0.4238 + 0.5306
#     =  0.9544 bits
#
#   CASE 5 — Information Gain (Decision Tree split)
#   ---------------------------------------------
#   Full dataset:  6 yes, 4 no   →  H = 0.971 bits (from Case 1)
#
#   Split by distance <10km (from Formula 05 applicant data):
#
#   Close group (n=6): 5 yes, 1 no
#     p = 5/6, 1/6
#     H_close = −(5/6)·log₂(5/6) − (1/6)·log₂(1/6)
#             = −(0.833)(−0.2630) − (0.167)(−2.5850)
#             = 0.2192 + 0.4308
#             = 0.6500 bits
#
#   Far group (n=4): 1 yes, 3 no
#     p = 1/4, 3/4
#     H_far  = −(0.25)·log₂(0.25) − (0.75)·log₂(0.75)
#            = −(0.25)(−2.0) − (0.75)(−0.4150)
#            = 0.5000 + 0.3113
#            = 0.8113 bits
#
#   Weighted child entropy:
#     (6/10)·0.6500 + (4/10)·0.8113
#     = 0.39 + 0.3245
#     = 0.7145 bits
#
#   Information Gain = H(parent) − weighted children
#                    = 0.9710 − 0.7145
#                    = 0.2565 bits
#
#   → Splitting by distance DROPPED entropy by 0.2565 bits.
#   → That is a good split. The Decision Tree would take it.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the counts list (number of samples per class).
#   NEVER edit the entropy calculation.
# ============================================================================


import math
import numpy as np


# ============================================================================
# PART 1: MANUAL ENTROPY — from scratch
# ============================================================================

print("=" * 68)
print("PART 1: SHANNON ENTROPY (manual, from scratch)")
print("=" * 68)


def shannon_entropy(counts):
    # Math:  H = −Σ p(xᵢ) · log₂(p(xᵢ))
    # Convention: 0 · log(0) = 0
    total = sum(counts)
    H = 0.0
    for c in counts:
        if c == 0:
            continue                        # 0 · log(0) = 0 by convention
        p = c / total
        H += p * math.log2(p)               # note: log2(p) is negative
    return -H                               # add the minus sign at the end


def information_gain(parent_counts, child_groups):
    # Math:  IG = H(parent) − Σ (|child|/|parent|) · H(child)
    H_parent = shannon_entropy(parent_counts)
    n_parent = sum(parent_counts)
    weighted = 0.0
    for child in child_groups:
        n_child = sum(child)
        weighted += (n_child / n_parent) * shannon_entropy(child)
    return H_parent - weighted


# ---------- CASE 1: 6 yes / 4 no ----------

print("\nCASE 1: 6 enrolled, 4 not enrolled")
H1 = shannon_entropy([6, 4])
print(f"  counts = [6, 4]")
print(f"  p = [0.600, 0.400]")
print(f"  H = {H1:.4f} bits   (near maximum 1.0)")


# ---------- CASE 2: 5 yes / 5 no (max) ----------

print("\nCASE 2: 5 enrolled, 5 not (50/50)")
H2 = shannon_entropy([5, 5])
print(f"  counts = [5, 5]")
print(f"  H = {H2:.4f} bits   ← MAXIMUM for 2 classes")


# ---------- CASE 3: 10 yes / 0 no (pure) ----------

print("\nCASE 3: 10 enrolled, 0 not (pure)")
H3 = shannon_entropy([10, 0])
print(f"  counts = [10, 0]")
print(f"  H = {H3:.4f} bits   ← MINIMUM (purity)")


# ---------- CASE 4: Jigjiga 2022 gender ----------

print("\nCASE 4: Jigjiga 2022 gender (50 male, 30 female)")
H4 = shannon_entropy([50, 30])
print(f"  counts = [50, 30]")
print(f"  p = [0.625, 0.375]")
print(f"  H = {H4:.4f} bits")


# ---------- CASE 5: Information Gain (Decision Tree split) ----------

print("\nCASE 5: Information Gain — split by distance")
parent = [6, 4]                             # 6 yes, 4 no
close  = [5, 1]                             # 5 yes, 1 no  (6 students)
far    = [1, 3]                             # 1 yes, 3 no  (4 students)

H_parent = shannon_entropy(parent)
H_close  = shannon_entropy(close)
H_far    = shannon_entropy(far)
IG       = information_gain(parent, [close, far])

print(f"  parent counts = [6, 4]     → H = {H_parent:.4f}")
print(f"  close  counts = [5, 1]     → H = {H_close:.4f}")
print(f"  far    counts = [1, 3]     → H = {H_far:.4f}")
print(f"  weighted child entropy     = "
      f"{(6/10)*H_close + (4/10)*H_far:.4f}")
print(f"  Information Gain           = {IG:.4f} bits")


# ============================================================================
# PART 2: LIBRARY VERSION — scipy.stats.entropy
# ============================================================================
#
#   scipy.stats.entropy(pk, base=2) takes probabilities and returns
#   Shannon entropy in the requested base (2 = bits).
#   Same formula. Faster. Production-ready.
#
# ============================================================================

from scipy.stats import entropy as scipy_entropy

print("\n" + "=" * 68)
print("PART 2: SHANNON ENTROPY (scipy library)")
print("=" * 68)


def probs(counts):
    # Convert counts to probabilities
    total = sum(counts)
    return [c / total for c in counts]


# ---------- Library: Cases 1–4 ----------

print("\nCASE 1: scipy H([0.6, 0.4]) =",
      f"{scipy_entropy(probs([6, 4]), base=2):.4f} bits")

print("CASE 2: scipy H([0.5, 0.5]) =",
      f"{scipy_entropy(probs([5, 5]), base=2):.4f} bits")

print("CASE 3: scipy H([1.0, 0.0]) =",
      f"{scipy_entropy(probs([10, 0]), base=2):.4f} bits")

print("CASE 4: scipy H([0.625, 0.375]) =",
      f"{scipy_entropy(probs([50, 30]), base=2):.4f} bits")


# ---------- Library: Information Gain ----------

print("\nCASE 5: Information Gain via scipy")
H_p = scipy_entropy(probs(parent), base=2)
H_c = scipy_entropy(probs(close),  base=2)
H_f = scipy_entropy(probs(far),    base=2)

n_p = sum(parent)
n_c = sum(close)
n_f = sum(far)

IG_lib = H_p - (n_c / n_p) * H_c - (n_f / n_p) * H_f

print(f"  H(parent) = {H_p:.4f}")
print(f"  H(close)  = {H_c:.4f}")
print(f"  H(far)    = {H_f:.4f}")
print(f"  IG        = {IG_lib:.4f} bits")


# ============================================================================
# PART 3: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Case 1 (6/4 split)     → H = {H1:.4f} bits  (mixed, hard)")
print(f"  Case 2 (5/5 split)     → H = {H2:.4f} bits  ← MAXIMUM for binary")
print(f"  Case 3 (10/0 pure)     → H = {H3:.4f} bits  ← MINIMUM (purity)")
print(f"  Case 4 (Jigjiga gender)→ H = {H4:.4f} bits")
print(f"  Case 5 (distance split)→ IG = {IG:.4f} bits (entropy dropped)")
print()
print("  WHAT THIS MEANS FOR JIGJIGA:")
print("  → Applicants are 60/40 YES/NO → entropy 0.97 (near max)")
print("  → The classification problem is HARD. Not easy.")
print("  → Splitting by distance dropped entropy by 0.26 bits")
print("  → That split is GOOD — a Decision Tree would take it")
print()
print("  WHAT THIS MEANS FOR WUCHALE:")
print("  → If we measure gender entropy in Wuchale pool:")
print("    20000M / 10000F → p = (0.667, 0.333) → H ≈ 0.918 bits")
print("  → Very mixed! No single strategy for males vs females")
print("  → We need multiple strategies — confirmed mathematically")
print()
print("  DECISION:")
print("  → Use entropy to pick WHICH feature to split on")
print("  → Highest Information Gain = best split = Decision Tree choice")
print("  → Cross-entropy loss in deep learning = same idea, different name")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - shannon_entropy() function
#      - information_gain() function
#      - scipy_entropy import / calls
#
#   🔁 ALWAYS edit:
#      - counts list (samples per class)
#      - for Information Gain: parent, child groups
#
#   ⚠️ counts list must contain non-negative integers
#   ⚠️ counts can include 0 — the convention 0·log(0)=0 handles it
#   ⚠️ For binary cases, entropy is capped at 1.0 (bits)
#   ⚠️ For k classes, entropy is capped at log₂(k)
#
# ============================================================================
# WHERE ENTROPY IS USED
# ============================================================================
#
#   Formula 05 (Logistic Regression)  → cross-entropy loss
#   Formula 09 (Neural Network)       → cross-entropy loss
#   Formula 12 (Decision Tree)        → information gain at every split
#   Formula 13 (Random Forest)        → many trees, all using entropy
#
# ============================================================================
