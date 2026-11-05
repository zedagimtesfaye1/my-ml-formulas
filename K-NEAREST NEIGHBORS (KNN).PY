# ============================================================================
# FORMULA 15: K-NEAREST NEIGHBORS (KNN)
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga
# Task   : Predict by looking at the K most SIMILAR past students
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "KNN is the laziest algorithm in machine learning — and one of the most
#  powerful. It does not learn a rule. It does not build a model. It does
#  not train. It simply REMEMBERS every past student.
#
#  To predict a new student, KNN asks ONE question: 'Who are the K most
#  SIMILAR students I have seen before?' Then it looks at their answers.
#  If most of them enrolled → the new student will enroll.
#  If most of them did NOT → they won't.
#
#  No equation. No tree. No weights. Just: LOOK AT THE NEIGHBORS.
#
#  Here is why it is powerful. Real life often works this way. A new
#  applicant from 11 km away, age 25 — who does she resemble? She
#  resembles the students who lived far and were older. Whatever THEY
#  did, she probably does too. KNN captures this intuition exactly.
#
#  For Jigjiga: 10 applicants on record. A new applicant arrives.
#  We look at the 3 most similar past applicants. Their votes decide.
#  That is KNN.
#
#  But there is a catch — and this is where it gets interesting. The
#  choice of K changes the answer. With K=3, we listen to 3 neighbors.
#  With K=7, we listen to 7. If the 3 nearest are NO but the wider
#  circle of 7 leans YES, the prediction FLIPS.
#
#  Small K = sensitive to noise, follows the closest few.
#  Large K = smooth, follows the crowd.
#  Choosing K is the entire art of KNN.
#
#  For Wuchale: KNN works perfectly — once 2027 data arrives. Because
#  KNN needs NO training, it needs only MEMORY. The moment we have
#  30,000 students from 2027, we can classify 2028 students by looking
#  at 2027's most similar. No re-training. No expensive GPU. Just
#  neighbors.
#
#  KNN is not sophisticated. It is honest. It says: 'Show me who
#  this person resembles. I will tell you what they did.' That is
#  how humans have always decided. KNN just puts it in math."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA — 3 STEPS
# ============================================================================
#
#   STEP 1 — DISTANCE between new point x and every training point xᵢ:
#
#       d(x, xᵢ) = √[ Σⱼ (xⱼ − xᵢⱼ)² ]
#
#       (Euclidean distance — one per training sample)
#
#   STEP 2 — FIND the K training points with the SMALLEST distance:
#
#       Nₖ(x) = the K nearest neighbors of x
#
#   STEP 3 — COMBINE their labels:
#
#       For CLASSIFICATION  (majority vote):
#           ŷ = MODE( yᵢ : xᵢ ∈ Nₖ(x) )
#
#       For REGRESSION  (average):
#           ŷ = (1/K) · Σ( yᵢ : xᵢ ∈ Nₖ(x) )
#
#       For WEIGHTED (closer neighbors count more):
#           ŷ = Σ( wᵢ · yᵢ ) / Σ( wᵢ )    where wᵢ = 1 / d(x, xᵢ)
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   x         = new point to classify (new applicant's features)
#   xᵢ        = i-th training point (a past applicant)
#   yᵢ        = label of past applicant (1 = YES, 0 = NO)
#   d(x, xᵢ)  = Euclidean distance between x and xᵢ
#   K         = number of neighbors to consider (chosen by us)
#   Nₖ(x)     = set of K nearest neighbors
#   MODE      = most frequent value (majority vote)
#   Σ         = sum over neighbors
#   wᵢ        = weight of neighbor i (for weighted KNN)
#   ŷ         = predicted label or value
#   √         = square root
#
# ============================================================================
# WHY IT MATTERS — KNN vs OTHER CLASSIFIERS
# ============================================================================
#
#   PROPERTY              | KNN                  | OTHERS
#   ----------------------|----------------------|---------------------
#   Training step         | NONE (instant)       | Expensive (fit)
#   Prediction step       | SLOW (scan all)      | Fast
#   Can overfit?          | YES (K=1)            | Tunable
#   Handles non-linear?   | ALWAYS (no assumptions) | Depends
#   Needs scaling?        | YES (distances)      | Usually
#   Interpretable?        | YES (show neighbors) | Sometimes
#   Handles missing?      | POORLY               | Depends
#   Small data?           | EXCELLENT            | Varies
#   Large data?           | SLOW                 | Usually fine
#
#   KNN is the model you use when:
#     - You have no idea what pattern exists (KNN assumes NOTHING)
#     - Dataset is small
#     - You want EXPLAINABLE predictions ("these 3 neighbors said...")
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA APPLICANT DATA
# ============================================================================
#
#   TRAINING DATA (10 past applicants):
#     #   Distance  Age   Registered
#     1      2      18      1  (YES)
#     2      5      22      1
#     3     15      25      0  (NO)
#     4      3      19      1
#     5     20      30      0
#     6      8      24      1
#     7     12      28      0
#     8      4      20      1
#     9      6      21      1
#     10    10      26      0
#
#   NEW APPLICANT:  Distance = 11,  Age = 25
#
#   ---------------------------------------------------------
#   STEP 1 — COMPUTE DISTANCES TO ALL 10 TRAINING POINTS
#   ---------------------------------------------------------
#   d = √[ (11 − distanceᵢ)² + (25 − ageᵢ)² ]
#
#   #   distance  age   d(i)
#   1      2      18   √(81 + 49) = √130  = 11.40
#   2      5      22   √(36 +  9) = √45   =  6.71
#   3     15      25   √(16 +  0) = √16   =  4.00
#   4      3      19   √(64 + 36) = √100  = 10.00
#   5     20      30   √(81 + 25) = √106  = 10.30
#   6      8      24   √( 9 +  1) = √10   =  3.16
#   7     12      28   √( 1 +  9) = √10   =  3.16
#   8      4      20   √(49 + 25) = √74   =  8.60
#   9      6      21   √(25 + 16) = √41   =  6.40
#   10    10      26   √( 1 +  1) = √2    =  1.41
#
#   ---------------------------------------------------------
#   STEP 2 — SORT BY DISTANCE (closest first)
#   ---------------------------------------------------------
#   Rank  #   distance  d(i)    label
#    1   10    10,26     1.41     0  (NO)
#    2    6     8,24     3.16     1  (YES)
#    3    7    12,28     3.16     0  (NO)      ← tie with #6
#    4    3    15,25     4.00     0  (NO)
#    5    9     6,21     6.40     1  (YES)
#    6    2     5,22     6.71     1  (YES)
#    7    8     4,20     8.60     1  (YES)
#    8    4     3,19    10.00     1  (YES)
#    9    5    20,30    10.30     0  (NO)
#   10    1     2,18    11.40     1  (YES)
#
#   ---------------------------------------------------------
#   STEP 3 — VOTE, DEPENDING ON K
#   ---------------------------------------------------------
#
#   K = 3  → neighbors are {10(NO), 6(YES), 7(NO)}
#            votes: 2 NO, 1 YES  →  PREDICT = NO
#
#   K = 5  → neighbors are {10(NO), 6(YES), 7(NO), 3(NO), 9(YES)}
#            votes: 3 NO, 2 YES  →  PREDICT = NO
#
#   K = 7  → neighbors are {10(NO), 6(YES), 7(NO), 3(NO), 9(YES),
#                          2(YES), 8(YES)}
#            votes: 3 NO, 4 YES  →  PREDICT = YES
#
#   → SAME applicant, SAME data, DIFFERENT K → DIFFERENT answer.
#   → K=3/5 say NO, K=7 says YES.
#   → This is why choosing K matters. There is no "correct" K
#     by math — only by cross-validation on real data.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the training data lists and the K value.
#   NEVER edit the distance or voting code.
#   All training lists MUST be same length.
#   ALWAYS standardize features if they have different units.
# ============================================================================


import numpy as np
import math
from collections import Counter


# ============================================================================
# PART 1: MANUAL KNN — from scratch
# ============================================================================

print("=" * 68)
print("PART 1: KNN (manual, from scratch)")
print("=" * 68)

# ---------- TRAINING DATA (ONLY EDIT THIS) ----------

distance   = [2, 5, 15, 3, 20, 8, 12, 4, 6, 10]
age        = [18, 22, 25, 19, 30, 24, 28, 20, 21, 26]
registered = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]

# ---------- NEW APPLICANT ----------

new_x = [11, 25]        # Distance = 11, Age = 25

# ---------- STEP 1: DISTANCES (NEVER EDIT) ----------

def euclidean(x1, x2):
    # Math:  d = √[ Σ (x1ⱼ − x2ⱼ)² ]
    return math.sqrt(sum((x1[j] - x2[j]) ** 2 for j in range(len(x1))))

distances = []
for i in range(len(distance)):
    pt = [distance[i], age[i]]
    d  = euclidean(pt, new_x)
    distances.append((d, registered[i], i + 1))

# ---------- STEP 2: SORT BY DISTANCE (NEVER EDIT) ----------

distances.sort(key=lambda t: t[0])

print(f"\nNew applicant: Distance={new_x[0]}, Age={new_x[1]}")
print(f"\nAll training points sorted by distance:")
print("  Rank  #   dist  age   d       label")
for r, (d, label, idx) in enumerate(distances):
    print(f"  {r+1:>3}  {idx:>2}  {distance[idx-1]:>3}   {age[idx-1]:>2}   "
          f"{d:5.2f}     {label}")


# ---------- STEP 3: PREDICT FOR DIFFERENT K ----------

def knn_predict(distances, K):
    # Math:  ŷ = MODE( labels of K nearest neighbors )
    neighbors = distances[:K]
    labels    = [n[1] for n in neighbors]
    vote      = Counter(labels).most_common(1)[0][0]
    return vote, labels


print("\nPredictions for different K:")
for K in [3, 5, 7]:
    pred, labels = knn_predict(distances, K)
    yes = labels.count(1)
    no  = labels.count(0)
    print(f"  K={K}:  neighbors labels = {labels}   "
          f"({yes} YES, {no} NO) → predict {'YES' if pred==1 else 'NO '}")


# ---------- WEIGHTED KNN (bonus) ----------
# Math:  ŷ = Σ(wᵢ·yᵢ) / Σ(wᵢ)  with  wᵢ = 1/dᵢ

def knn_weighted(distances, K):
    neighbors = distances[:K]
    weight_yes = 0.0
    weight_no  = 0.0
    for d, label, _ in neighbors:
        w = 1.0 / (d + 1e-9)      # avoid division by zero
        if label == 1:
            weight_yes += w
        else:
            weight_no += w
    total = weight_yes + weight_no
    return 1 if weight_yes >= weight_no else 0, weight_yes / total

print("\nWeighted KNN (closer neighbors count more):")
for K in [3, 5, 7]:
    pred, prob = knn_weighted(distances, K)
    print(f"  K={K}:  P(YES) = {prob:.3f} → predict "
          f"{'YES' if pred==1 else 'NO '}")


# ============================================================================
# PART 2: LIBRARY KNN — scikit-learn KNeighborsClassifier
# ============================================================================
#
#   sklearn handles distance, sorting, voting — all in one line.
#   Same math. Faster. Production-ready.
#
# ============================================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("\n" + "=" * 68)
print("PART 2: KNN (scikit-learn)")
print("=" * 68)

# ---------- BUILD X, Y ----------
X_train = np.array([[distance[i], age[i]] for i in range(len(distance))], dtype=float)
Y_train = np.array(registered, dtype=int)

# ---------- STANDARDIZE (KNN is distance-based — scaling matters!) ----------
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
new_scaled = scaler.transform([new_x])

# ---------- TRAIN + PREDICT FOR DIFFERENT K ----------

print(f"\nNew applicant: Distance={new_x[0]}, Age={new_x[1]}")
print("\nsklearn predictions:")
for K in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_scaled, Y_train)
    pred  = knn.predict(new_scaled)[0]
    proba = knn.predict_proba(new_scaled)[0]
    print(f"  K={K}:  predict = {'YES' if pred==1 else 'NO '}   "
          f"P(NO)={proba[0]:.2f}  P(YES)={proba[1]:.2f}")

# ---------- FIND BEST K VIA ACCURACY ON TRAINING ----------

print("\nFinding best K (accuracy on training data):")
for K in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_scaled, Y_train)
    acc = accuracy_score(Y_train, knn.predict(X_scaled)) * 100
    print(f"  K={K}:  training accuracy = {acc:.1f}%")

# ---------- SHOW NEIGHBORS ----------

print("\nThe 5 nearest neighbors (K=5) for the new applicant:")
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_scaled, Y_train)
neighbor_indices = knn5.kneighbors(new_scaled, return_distance=False)[0]

for idx in neighbor_indices:
    print(f"  Applicant #{idx+1}: dist={distance[idx]:>2} age={age[idx]:>2} "
          f"label={'YES' if registered[idx]==1 else 'NO '}")


# ============================================================================
# PART 3: KNN REGRESSOR — predict numbers, not classes
# ============================================================================
#
#   For Jigjiga enrollment (a NUMBER), use KNeighborsRegressor.
#   Same idea: predict = AVERAGE of neighbors' values.
#
# ============================================================================

from sklearn.neighbors import KNeighborsRegressor

print("\n" + "=" * 68)
print("PART 3: KNN REGRESSOR (Jigjiga enrollment)")
print("=" * 68)

year_col   = [0, 1, 2, 3, 4]
female_col = [30, 40, 30, 50, 20]
total_col  = [80, 110, 50, 130, 30]

X_reg = np.array([[year_col[i], female_col[i]] for i in range(5)], dtype=float)
Y_reg = np.array(total_col, dtype=float)

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

knn_reg = KNeighborsRegressor(n_neighbors=2)   # K=2 (only 5 points)
knn_reg.fit(X_reg_scaled, Y_reg)

preds_reg = knn_reg.predict(X_reg_scaled)
rmse_reg = np.sqrt(np.mean((preds_reg - Y_reg) ** 2))

print(f"\nKNN Regressor training RMSE = {rmse_reg:.2f}   "
      f"(Multiple Reg was 8.48)")

# Future predictions (KNN CANNOT extrapolate — same limit as trees)
future = [[5, 31], [6, 30]]
future_scaled = scaler_reg.transform(future)
future_preds  = knn_reg.predict(future_scaled)

print("\nFuture predictions (⚠ no true extrapolation):")
for j in range(2):
    print(f"  {2027+j} (year={future[j][0]}, female={future[j][1]}) "
          f"→ {future_preds[j]:.1f}")


# ============================================================================
# PART 4: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  K=3 manual → NO    (2 NO, 1 YES)")
print(f"  K=5 manual → NO    (3 NO, 2 YES)")
print(f"  K=7 manual → YES   (3 NO, 4 YES)   ← same data, different K!")
print()
print("  MEANING:")
print("  → KNN does not learn rules. It remembers and looks at neighbors.")
print("  → K matters: small K = follows the closest few; large K = the crowd.")
print("  → K=3 and K=5 say NO, K=7 says YES — same applicant.")
print("  → No magic correct K — use cross-validation to choose.")
print()
print("  JIGJIGA DECISION:")
print(f"  → New applicant (dist=11, age=25) → majority vote says NO")
print(f"  → Closest neighbor is applicant #10 (dist=10, age=26) who said NO.")
print(f"  → So: CALL this applicant. They resemble our NO-shows.")
print()
print("  WUCHALE NOTE:")
print("  → KNN needs no training — only past data. Feed 2027 real records")
print("    in 2028 and immediately classify new applicants.")
print("  → For Wuchale 2027 (no history), use K-Means (Formula 07).")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - euclidean(), knn_predict(), knn_weighted() functions
#      - KNeighborsClassifier / KNeighborsRegressor construction
#      - StandardScaler() fitting
#
#   🔁 ALWAYS edit:
#      - distance, age, registered lists (classification)
#      - year_col, female_col, total_col (regression)
#      - new_x  (the point to predict)
#      - K  (try 3, 5, 7, 9 — check accuracy per K)
#
#   ⚙️ AUTO-computed:
#      - all distances
#      - sorted neighbors
#      - majority vote
#      - weighted probabilities
#
#   ⚠️ ALWAYS standardize features first — distance is scale-sensitive
#   ⚠️ Small K (1, 3) = noisy, overfit
#   ⚠️ Large K = smooth, but can blur real boundaries
#   ⚠️ K must be ODD for binary classes (avoids ties)
#   ⚠️ KNN cannot extrapolate — for future years, use regression
#
# ============================================================================
# WHEN TO USE KNN
# ============================================================================
#
#   ✅ USE IT WHEN:
#     - Small dataset (< 10,000 rows)
#     - No assumption about pattern shape
#     - You want EXPLAINABLE predictions ("these 3 neighbors said...")
#     - You have many features but few samples
#
#   ❌ DON'T USE IT WHEN:
#     - Very large dataset (prediction is slow — O(n) per prediction)
#     - Features have wildly different scales (must standardize)
#     - You need to extrapolate beyond training data
#     - High dimension (curse of dimensionality — distances lose meaning)
#
# ============================================================================
