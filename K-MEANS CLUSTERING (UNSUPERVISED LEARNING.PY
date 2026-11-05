# ============================================================================
# FORMULA 07: K-MEANS CLUSTERING  (UNSUPERVISED LEARNING)
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Wuchale (new area, 2027 launch)
# Task   : Group students WITHOUT labels — the first unsupervised formula
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Every formula until today needed LABELS. To predict students, we
#  needed to already KNOW past answers. But Wuchale has NO past.
#  It's a NEW area. We have 30,000 potential students and ZERO history.
#  Supervised learning is blind here. It cannot help.
#
#  Today we meet the formula that needs NO labels. It's called K-MEANS.
#  It is UNSUPERVISED — it discovers patterns on its own.
#
#  Here's what it does. Look at the 30,000 students of Wuchale. Don't
#  tell the model anything about who is who. Just show it WHERE they
#  live and HOW OLD they are. K-Means will find the natural groups
#  by itself.
#
#  For Wuchale, it will discover TWO groups:
#     Cluster 1 — students CLOSE to town, YOUNG (typical age)
#     Cluster 2 — students FAR from town, OLDER (working, married)
#
#  NOBODY told the model about 'close' or 'far' or 'young' or 'old.'
#  It discovered those groups from the data alone.
#
#  Now we know what to do:
#    Cluster 1 — normal registration in town
#    Cluster 2 — MOBILE registration center. Evening hours. Transport.
#
#  This is unsupervised learning. It doesn't predict. It REVEALS.
#  And for a new area like Wuchale, revelation is exactly what we need."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA
# ============================================================================
#
#   STEP 1 — Distance between point x and centroid c (Euclidean):
#
#       d(x, c) = √[ (x₁−c₁)² + (x₂−c₂)² + ... + (xₚ−cₚ)² ]
#
#   STEP 2 — ASSIGN each point to its nearest centroid:
#
#       cluster(x) = argmin over k of d(x, cₖ)
#
#   STEP 3 — UPDATE each centroid to be the mean of its cluster:
#
#       cₖ = (1 / |Sₖ|) · Σ over x in Sₖ of x
#
#       where Sₖ = set of points assigned to cluster k
#
#   STEP 4 — REPEAT steps 2 and 3 until centroids stop moving (convergence).
#
# SYMBOLS:
#   x   = one data point (a student's features)
#   cₖ  = centroid of cluster k (the "center")
#   d   = Euclidean distance
#   K   = number of clusters (we choose: K = 2 for Wuchale)
#   Sₖ  = set of points in cluster k
#   |Sₖ| = size of cluster k
#   p   = number of features (2 here: distance, age)
#   Σ   = sum
#   √   = square root
#   argmin = "the k that gives the smallest value"
#
# ============================================================================
# WHY IT MATTERS — SUPERVISED vs UNSUPERVISED
# ============================================================================
#
#   FEATURE          | SUPERVISED (F01-F05)     | UNSUPERVISED (F07)
#   -----------------|--------------------------|----------------------
#   Needs labels?    | YES (needs past answers) | NO  (no labels)
#   Can predict?     | Yes (predict next year)  | No (finds groups)
#   Wuchale support? | ❌ NO HISTORY            | ✅ WORKS
#   Output           | Number or Yes/No         | CLUSTERS (groups)
#   Example use      | "56 students in 2027"    | "2 groups of students"
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT WUCHALE STUDENT SAMPLE
# ============================================================================
#
#   Wuchale sample: 6 students (scaled down from 30,000 for clarity)
#   Features per student: (distance_km, age)
#
#     #   Distance   Age     (for clarity, note)
#     --  --------   ----    -----------------
#     P1     2        16     close, young
#     P2     3        17     close, young
#     P3     5        15     close, young
#     P4    15        26     far, older
#     P5    18        28     far, older
#     P6    20        25     far, older
#
#   We choose K = 2 clusters.
#
#   INITIAL CENTROIDS (chosen as P1 and P6 for clarity):
#     c₁ = (2, 16)
#     c₂ = (20, 25)
#
#   ---------------------------------------------
#   ITERATION 1
#   ---------------------------------------------
#
#   Assign each point to nearest centroid:
#     P1(2,16):  d to c₁=√(0²+0²)=0       d to c₂=√(18²+9²)=20.12 → c₁
#     P2(3,17):  d to c₁=√(1+1)=1.41      d to c₂=√(17²+8²)=18.79 → c₁
#     P3(5,15):  d to c₁=√(9+1)=3.16      d to c₂=√(15²+10²)=18.03 → c₁
#     P4(15,26): d to c₁=√(169+100)=16.40 d to c₂=√(25+1)=5.10   → c₂
#     P5(18,28): d to c₁=√(256+144)=20.00 d to c₂=√(4+9)=3.61    → c₂
#     P6(20,25): d to c₁=√(324+81)=20.12  d to c₂=0              → c₂
#
#   Cluster 1 = {P1, P2, P3}
#   Cluster 2 = {P4, P5, P6}
#
#   NEW CENTROIDS (mean of each cluster):
#     c₁ = ((2+3+5)/3, (16+17+15)/3) = (3.33, 16.00)
#     c₂ = ((15+18+20)/3, (26+28+25)/3) = (17.67, 26.33)
#
#   ---------------------------------------------
#   ITERATION 2 (with new centroids)
#   ---------------------------------------------
#
#     P1(2,16):  d to c₁=√(1.77+0)=1.33    d to c₂=√(245.5+106.7)=18.77 → c₁
#     P2(3,17):  d to c₁=√(0.11+1)=1.05    d to c₂=√(215.2+86.5)=17.37 → c₁
#     P3(5,15):  d to c₁=√(2.79+1)=1.95    d to c₂=√(160.5+128.2)=16.99 → c₁
#     P4(15,26): d to c₁=√(136.2+100)=15.37 d to c₂=√(7.13+0.11)=2.69 → c₂
#     P5(18,28): d to c₁=√(215.2+144)=18.95 d to c₂=√(0.11+2.79)=1.70 → c₂
#     P6(20,25): d to c₁=√(277.9+81)=18.95 d to c₂=√(5.43+1.77)=2.68 → c₂
#
#   Clusters SAME as before → CONVERGED. Stop.
#
#   ---------------------------------------------
#   FINAL RESULT
#   ---------------------------------------------
#
#   Cluster 1 (CLOSE + YOUNG):   P1, P2, P3    centroid = (3.33, 16.0)
#   Cluster 2 (FAR + OLDER):     P4, P5, P6    centroid = (17.67, 26.33)
#
#   ---------------------------------------------
#   SCALE UP TO 30,000 WUCHALE STUDENTS
#   ---------------------------------------------
#
#   Same algorithm. Same formula. Just feed all 30,000.
#   Result: same 2 clusters discovered (or 3-4 if K is bigger).
#   Each cluster = one REGISTRATION STRATEGY.
#
#   Cluster "CLOSE + YOUNG"   → normal registration in town
#   Cluster "FAR + OLDER"     → MOBILE registration + evening hours
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the data lists (distance, age) and K.
#   NEVER edit the distance function or iteration loop.
#   All feature lists MUST have the same length.
#   NOTE: NO Y labels needed — that is the point of unsupervised learning.
# ============================================================================


# ============================================================================
# PART 1: MANUAL VERSION — from scratch (transparent math)
# ============================================================================
#
#   Install once:  pip install numpy
#
# ============================================================================

import math
import numpy as np

print("=" * 68)
print("FORMULA 07: K-MEANS CLUSTERING  (UNSUPERVISED)")
print("=" * 68)

# ---------- STEP 1: YOUR DATA (ONLY EDIT THIS SECTION) ----------
# Wuchale sample students. Features: [distance_km, age]
# NO Y LABELS. That's the whole point.

students = [
    [2,  16],   # P1
    [3,  17],   # P2
    [5,  15],   # P3
    [15, 26],   # P4
    [18, 28],   # P5
    [20, 25],   # P6
]

K = 2   # number of clusters (2 for Wuchale: close/young vs far/older)

# Initial centroids — for clarity, chosen as first and last points.
# In practice, random initialization is fine (library does this automatically).
centroids = [
    [2, 16],     # c₁ = starting at P1
    [20, 25],    # c₂ = starting at P6
]

# ---------- STEP 2: EUCLIDEAN DISTANCE (NEVER EDIT) ----------
# Math:  d(x,c) = √[ (x₁−c₁)² + (x₂−c₂)² + ... ]

def distance(point, centroid):
    total = 0
    for j in range(len(point)):
        diff = point[j] - centroid[j]
        total = total + diff * diff
    return math.sqrt(total)

# ---------- STEP 3: K-MEANS LOOP (NEVER EDIT) ----------
# Math:
#   assign each point to nearest centroid
#   update each centroid to the mean of its cluster
#   repeat until centroids stop moving

print("\n--- MANUAL VERSION ---")

for iteration in range(100):     # max 100 to avoid infinite loop
    # --- Assignment step ---
    clusters = [[] for _ in range(K)]   # clusters[k] = list of point indices
    for i, point in enumerate(students):
        best_k = 0
        best_d = distance(point, centroids[0])
        for k in range(1, K):
            d = distance(point, centroids[k])
            if d < best_d:
                best_d = d
                best_k = k
        clusters[best_k].append(i)

    # --- Update step ---
    new_centroids = []
    for k in range(K):
        if len(clusters[k]) == 0:
            new_centroids.append(centroids[k])   # keep old if empty
            continue
        sums = [0.0] * len(students[0])
        for idx in clusters[k]:
            for j in range(len(students[0])):
                sums[j] = sums[j] + students[idx][j]
        mean = [s / len(clusters[k]) for s in sums]
        new_centroids.append(mean)

    # --- Check convergence ---
    moved = 0.0
    for k in range(K):
        moved = moved + distance(new_centroids[k], centroids[k])
    centroids = new_centroids

    print(f"Iteration {iteration+1}: centroids = "
          f"{[tuple(round(v,2) for v in c) for c in centroids]}  "
          f"moved = {round(moved, 4)}")

    if moved < 0.0001:
        print("Converged. Stop.")
        break

# ---------- STEP 4: SHOW FINAL GROUPS (NEVER EDIT) ----------

print("\nFinal groups:")
for k in range(K):
    print(f"\n  Cluster {k+1}  (centroid = {tuple(round(v,2) for v in centroids[k])}):")
    for idx in clusters[k]:
        print(f"    P{idx+1}: distance={students[idx][0]}  age={students[idx][1]}")


# ============================================================================
# PART 2: LIBRARY VERSION — scikit-learn (production code)
# ============================================================================
#
#   Install once:  pip install scikit-learn
#   Same math inside. Two lines for training + prediction.
#
# ============================================================================

from sklearn.cluster import KMeans

print("\n" + "=" * 68)
print("LIBRARY VERSION (scikit-learn KMeans)")
print("=" * 68)

X = np.array(students)

model = KMeans(n_clusters=K, random_state=42, n_init=10)
model.fit(X)

labels    = model.labels_              # cluster ID per student (0 or 1)
centers   = model.cluster_centers_     # final centroids
inertia   = model.inertia_             # sum of squared distances to centers

print("\nFinal groups:")
for k in range(K):
    print(f"\n  Cluster {k+1}  (centroid = "
          f"{tuple(round(v,2) for v in centers[k])}):")
    for i in range(len(students)):
        if labels[i] == k:
            print(f"    P{i+1}: distance={students[i][0]}  age={students[i][1]}")

print(f"\nInertia (lower = tighter clusters): {inertia:.2f}")


# ============================================================================
# PART 3: FINAL SUMMARY — WUCHALE STRATEGY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY — WUCHALE 2027 STRATEGY")
print("=" * 68)
print("  Cluster 1 = CLOSE + YOUNG  → normal registration in town")
print("  Cluster 2 = FAR + OLDER    → mobile registration + evening hours")
print()
print("  SCALE UP: apply same K-Means to all 30,000 Wuchale students.")
print("  Each cluster tells you ONE registration strategy.")
print()
print("  WHAT YOU SAY IN THE MEETING:")
print("  → 'We have 2 clear groups in Wuchale:')
print("     one reachable in town, one needs mobile service.")
print("     Two strategies, not one. This doubles our coverage.'")
print()
print("  HOW TO CHOOSE K (how many clusters):")
print("  → Try K=2, 3, 4 — plot inertia (elbow method).")
print("  → Where curve bends = the right K.")
print("  → Library computes inertia for you automatically.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - distance() function
#      - the K-Means assignment/update loop
#      - sklearn KMeans() call
#
#   🔁 ALWAYS edit:
#      - students list (add more points, more features)
#      - K (how many clusters you want)
#      - All feature lists MUST be same length per point
#
#   ⚙️ AUTO-computed:
#      - centroids
#      - cluster assignments
#      - inertia
#
#   ⚠️ NO Y LABELS — that is the whole point of unsupervised learning
#
# ============================================================================
# WUCHALE — SOLVED HERE ✅
# ============================================================================
#   This is the formula Wuchale needed. No history required.
#   Feed the 30,000 student pool → get strategy groups.
#   Next formulas (Neural Network) will also work unsupervised,
#   but K-Means is simpler and sufficient for strategy planning.
# ============================================================================
