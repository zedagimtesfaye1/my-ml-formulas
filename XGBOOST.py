# ============================================================================
# FORMULA 13: XGBOOST / GRADIENT BOOSTING + OBJECTIVE FUNCTION
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga
# Task   : The most powerful algorithm on tabular data — beats RF, beats NN
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Random Forest builds trees in PARALLEL — each one blind to the others.
#  Then they vote. Good. But not the best.
#
#  Gradient Boosting builds trees SEQUENTIALLY. Each tree is trained to
#  fix the MISTAKES of the previous trees. Not to vote. To CORRECT.
#
#  Imagine 100 doctors, one after another. Doctor 1 makes a diagnosis.
#  Doctor 2 reads the diagnosis, finds what Doctor 1 got WRONG, and
#  corrects it. Doctor 3 corrects Doctor 2. And so on. After 100 doctors,
#  the diagnosis is almost perfect. THAT is gradient boosting.
#
#  XGBoost ('Extreme Gradient Boosting') is the best-known version of
#  this idea. It won more Kaggle competitions than any other algorithm.
#  It is the default workhorse on tabular data — spreadsheets, CSVs,
#  business data. It beats Random Forest by 5-10%. Often beats neural
#  networks on table data.
#
#  What makes XGBoost special is its OBJECTIVE FUNCTION. That is the
#  math formula at the heart of this file. It has two parts:
#    LOSS    = how wrong is the model?
#    PENALTY = how complex is the model?
#  XGBoost optimizes BOTH. It wants to be accurate AND simple. That is
#  why it does not overfit like a single tree.
#
#  For Jigjiga: instead of 100 trees voting, we get 100 trees CORRECTING
#  each other, one after another. Predictions are sharper. Errors shrink
#  iteration by iteration.
#
#  For Wuchale: same story. No history → cannot train. But once 2027
#  data arrives, XGBoost is the FIRST algorithm to try. It will handle
#  missing data, mixed features, and outliers better than anything else.
#
#  XGBoost is not magic. It is one idea, done 100 times: CORRECT THE
#  PREVIOUS MISTAKE. And that simple idea is the strongest thing we
#  have on tabular data."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA — THE OBJECTIVE FUNCTION
# ============================================================================
#
#   THE XGBOOST OBJECTIVE (what we minimize):
#
#       Obj = Σᵢ L(yᵢ, ŷᵢ)   +   Σₖ Ω(fₖ)
#             \_____/            \_____/
#             training loss      regularization penalty
#
#   where:
#       L(yᵢ, ŷᵢ)    = loss on sample i (e.g., (yᵢ − ŷᵢ)²  for MSE)
#       Ω(fₖ)        = complexity of tree k
#                    = γ·T + ½·λ·Σⱼ wⱼ²
#                    where T = number of leaves in tree k
#                          wⱼ = weight (score) of leaf j
#                          γ  = penalty per leaf
#                          λ  = L2 penalty on leaf weights
#
#   At iteration t, we ADD a new tree fₜ. Objective becomes:
#
#       Obj⁽ᵗ⁾ = Σᵢ L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ))  +  Ω(fₜ)  +  const
#
#   We approximate the loss with a 2nd-order TAYLOR expansion:
#
#       Obj⁽ᵗ⁾ ≈ Σᵢ [ gᵢ·fₜ(xᵢ)  +  ½·hᵢ·fₜ²(xᵢ) ]  +  Ω(fₜ)
#
#   where  gᵢ = ∂L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) / ∂ŷᵢ         ← gradient (first deriv)
#          hᵢ = ∂²L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) / ∂ŷᵢ²        ← hessian (second deriv)
#
#   For MSE with L = ½(y − ŷ)²:
#       gᵢ = ŷᵢ − yᵢ              ← residual (predicted − actual)
#       hᵢ = 1                     ← constant
#
#   Now group samples by leaf. For a leaf with sample set Iⱼ:
#
#       Gⱼ = Σᵢ∈Iⱼ gᵢ              ← sum of gradients in leaf j
#       Hⱼ = Σᵢ∈Iⱼ hᵢ              ← sum of hessians in leaf j
#
#   Optimal leaf weight (minimize Obj⁽ᵗ⁾ w.r.t. wⱼ):
#
#       wⱼ* = − Gⱼ / (Hⱼ + λ)
#
#   Optimal objective value for leaf j:
#
#       Objⱼ* = − ½ · Gⱼ² / (Hⱼ + λ)
#
#   The GAIN from a split (left L, right R):
#
#       Gain = ½ · [ G_L²/(H_L+λ)  +  G_R²/(H_R+λ)  −  (G_L+G_R)²/(H_L+H_R+λ) ]  −  γ
#
#   → choose the split with the LARGEST Gain.
#
# ============================================================================
# SYMBOLS
# ============================================================================
#
#   Obj          = objective function (what XGBoost minimizes)
#   L            = loss (training error)
#   Ω            = regularization (complexity penalty)
#   ŷᵢ           = current prediction for sample i
#   yᵢ           = true target for sample i
#   fₜ           = the new tree added at iteration t
#   gᵢ           = gradient ∂L/∂ŷᵢ
#   hᵢ           = hessian ∂²L/∂ŷᵢ²
#   wⱼ           = weight (output value) of leaf j
#   T            = number of leaves
#   Gⱼ, Hⱼ       = sum of gradients / hessians in leaf j
#   γ (gamma)    = penalty per leaf (larger γ → simpler trees)
#   λ (lambda)   = L2 penalty on leaf weights (larger λ → smaller weights)
#   η (eta)      = learning rate (shrinkage); typically 0.01–0.3
#   n_estimators = number of trees (= T iterations)
#
# ============================================================================
# WHY IT MATTERS — RANDOM FOREST vs GRADIENT BOOSTING
# ============================================================================
#
#   PROPERTY              | RANDOM FOREST        | GRADIENT BOOSTING (XGB)
#   ----------------------|----------------------|------------------------
#   How trees combine     | PARALLEL (vote)      | SEQUENTIAL (correct)
#   What each tree does   | Independent guess    | Fix previous errors
#   Typical accuracy      | 85–92%               | 90–97%
#   Overfitting risk      | Very low             | Low (with regularization)
#   Speed                 | Fast                 | Fast (XGBoost optimized)
#   Kaggle winners        | Many                 | MORE
#   Tabular data          | Excellent            | Excellent+
#   Needs tuning?         | Almost none          | Yes (λ, γ, η, depth)
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA DATA
# ============================================================================
#
#   DATA (total enrollment 2022–2026):
#     Year   Y (actual total)
#       0        80
#       1        110
#       2        50
#       3        130
#       4        30
#
#   ---------------------------------------------------------
#   ITERATION 0 — START FROM A CONSTANT
#   ---------------------------------------------------------
#   ŷᵢ⁽⁰⁾ = mean(Y) = (80+110+50+130+30)/5 = 80 for ALL samples.
#
#   ---------------------------------------------------------
#   ITERATION 1 — COMPUTE GRADIENTS + HESSIANS (MSE loss)
#   ---------------------------------------------------------
#   For L = ½(yᵢ − ŷᵢ)²:
#     gᵢ = ŷᵢ − yᵢ
#     hᵢ = 1
#
#   Sample  yᵢ   ŷᵢ⁽⁰⁾    gᵢ = ŷᵢ − yᵢ    hᵢ
#     0     80    80          0           1
#     1    110    80        −30           1
#     2     50    80        +30           1
#     3    130    80        −50           1
#     4     30    80        +50           1
#
#   ---------------------------------------------------------
#   ITERATION 1 — TRY ONE SPLIT:  year ≤ 3  vs  year = 4
#   ---------------------------------------------------------
#   LEFT leaf  (samples 0,1,2,3):
#     G_L = 0 + (−30) + 30 + (−50) = −50
#     H_L = 1 + 1 + 1 + 1 = 4
#
#   RIGHT leaf (sample 4):
#     G_R = +50
#     H_R = 1
#
#   Suppose λ = 1, γ = 0.
#
#   Optimal leaf weights:
#     w_L* = −G_L / (H_L + λ) = 50 / (4 + 1) = +10.0
#     w_R* = −G_R / (H_R + λ) = −50 / (1 + 1) = −25.0
#
#   Gain from this split:
#     Gain = ½ · [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ
#          = ½ · [ 2500/5 + 2500/2 − 0²/5 ] − 0
#          = ½ · [ 500 + 1250 − 0 ]
#          = 875
#
#   → Gain = 875. This split is VERY GOOD.
#
#   ---------------------------------------------------------
#   ITERATION 1 — UPDATE PREDICTIONS (with learning rate η = 0.3)
#   ---------------------------------------------------------
#   LEFT branch  (years 0,1,2,3):  ŷ₁ = 80 + 0.3 × (+10) = 83.0
#   RIGHT branch (year 4):         ŷ₁ = 80 + 0.3 × (−25) = 72.5
#
#   New errors:
#     Year 0:  80 − 83.0    = −3.0   (was 0 at iter 0)
#     Year 1:  110 − 83.0   = +27.0  (was +30 — smaller!)
#     Year 2:  50 − 83.0    = −33.0  (was −30 — slightly bigger here)
#     Year 3:  130 − 83.0   = +47.0  (was +50 — smaller!)
#     Year 4:  30 − 72.5    = −42.5  (was −50 — smaller!)
#
#   Total absolute error:
#     Iteration 0: 0 + 30 + 30 + 50 + 50 = 160
#     Iteration 1: 3 + 27 + 33 + 47 + 42.5 = 152.5   ← improved
#
#   ONE tree helped. 100 trees will do much more.
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit the feature lists and target.
#   NEVER edit the objective function or the library call.
# ============================================================================


import numpy as np
from collections import Counter


# ============================================================================
# PART 1: MANUAL GRADIENT BOOSTING — from scratch (2 trees)
# ============================================================================

print("=" * 68)
print("PART 1: GRADIENT BOOSTING (manual, from scratch)")
print("=" * 68)

# ---------- DATA (ONLY EDIT THIS) ----------

years  = [0, 1, 2, 3, 4]
total  = [80, 110, 50, 130, 30]

n = len(total)
Y = np.array(total, dtype=float)

# ---------- HYPERPARAMETERS (CHOSEN for illustration) ----------

eta  = 0.3      # learning rate
lam  = 1.0      # L2 penalty (lambda)
gam  = 0.0      # leaf penalty (gamma)

# ---------- STEP 1: INITIAL PREDICTION = MEAN ----------

base = np.mean(Y)
yhat = np.array([base] * n)                # ŷ⁽⁰⁾ = base for all

print(f"\nBase prediction = mean = {base:.2f}")
print(f"Initial predictions: {yhat}")

# ---------- STEP 2: ONE SPLIT (hardcoded for clarity) ----------
# Split: year ≤ 3 → left leaf  |  year = 4 → right leaf

def split_indices(year_value):
    if year_value <= 3:
        return "L"
    else:
        return "R"

# ---------- STEP 3: COMPUTE g, h, G, H ----------

g = yhat - Y                                # gradient for MSE
h = np.ones(n)                              # hessian = 1

G_L = sum(g[i] for i in range(n) if split_indices(years[i]) == "L")
G_R = sum(g[i] for i in range(n) if split_indices(years[i]) == "R")
H_L = sum(h[i] for i in range(n) if split_indices(years[i]) == "L")
H_R = sum(h[i] for i in range(n) if split_indices(years[i]) == "R")

print(f"\nIteration 1:")
print(f"  Gradients g = {g}")
print(f"  G_L = {G_L:+.2f}   H_L = {H_L:.2f}")
print(f"  G_R = {G_R:+.2f}   H_R = {H_R:.2f}")

# ---------- STEP 4: OPTIMAL LEAF WEIGHTS ----------

w_L = -G_L / (H_L + lam)
w_R = -G_R / (H_R + lam)

print(f"\n  w_L* = -G_L/(H_L+λ) = {-G_L:.2f}/{H_L+lam:.2f} = {w_L:+.2f}")
print(f"  w_R* = -G_R/(H_R+λ) = {-G_R:.2f}/{H_R+lam:.2f} = {w_R:+.2f}")

# ---------- STEP 5: GAIN ----------

gain = 0.5 * (G_L**2 / (H_L + lam)
            + G_R**2 / (H_R + lam)
            - (G_L + G_R)**2 / (H_L + H_R + lam)) - gam

print(f"\n  Gain = ½·[ {G_L**2:.0f}/{H_L+lam:.0f} "
      f"+ {G_R**2:.0f}/{H_R+lam:.0f} "
      f"− {(G_L+G_R)**2:.0f}/{H_L+H_R+lam:.0f} ] − γ")
print(f"       = {gain:.2f}")

# ---------- STEP 6: UPDATE PREDICTIONS ----------

for i in range(n):
    side = split_indices(years[i])
    leaf_w = w_L if side == "L" else w_R
    yhat[i] = yhat[i] + eta * leaf_w

print(f"\n  Updated predictions (η={eta}):")
for i in range(n):
    print(f"    Year={years[i]}  actual={total[i]}  predicted={yhat[i]:.2f}")

mae = np.mean(np.abs(yhat - Y))
print(f"\n  MAE after iteration 1 = {mae:.2f}")
print(f"  MAE before (base)     = {np.mean(np.abs(base - Y)):.2f}")


# ============================================================================
# PART 2: LIBRARY — sklearn GradientBoostingRegressor
# ============================================================================
#
#   sklearn's GradientBoostingRegressor implements the SAME objective
#   function. Just faster and with hundreds of trees.
#
# ============================================================================

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("\n" + "=" * 68)
print("PART 2: GRADIENT BOOSTING (sklearn)")
print("=" * 68)

X_sk = np.array([[years[i]] for i in range(n)], dtype=float)

model = GradientBoostingRegressor(
    n_estimators=100,       # 100 trees, each correcting the last
    learning_rate=0.1,      # η
    max_depth=3,            # shallow trees = "weak learners"
    random_state=42,
)
model.fit(X_sk, Y)

preds_sk = model.predict(X_sk)
rmse_sk = np.sqrt(mean_squared_error(Y, preds_sk))

print(f"\n  Training predictions:")
for i in range(n):
    print(f"    Year={years[i]}  actual={total[i]}  predicted={preds_sk[i]:.2f}")

print(f"\n  Training RMSE = {rmse_sk:.2f}   (Linear Reg was 35.1)")

# ---------- EXTRAPOLATION WARNING ----------
# Trees CANNOT extrapolate beyond training range. year=5 (2027) is
# outside [0,4] — so prediction will fall back to nearest seen value.

print("\n  Extrapolation warning:")
print("  → Trees CANNOT predict beyond training years (0–4).")
print("  → For 2027 (year=5), the model outputs the nearest seen value.")

future = np.array([[5], [6]], dtype=float)
future_preds = model.predict(future)
for j in range(2):
    print(f"    2027+j (year={int(future[j][0])}) → {future_preds[j]:.2f}  "
          f"(⚠ not true extrapolation)")


# ============================================================================
# PART 3: LIBRARY — XGBoost (if installed)
# ============================================================================
#
#   Install once:  pip install xgboost
#   XGBoost is faster and has more knobs than sklearn's version.
#   The math is IDENTICAL — same objective function shown above.
#
# ============================================================================

try:
    from xgboost import XGBRegressor

    print("\n" + "=" * 68)
    print("PART 3: XGBOOST (xgboost library)")
    print("=" * 68)

    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        reg_lambda=1.0,      # λ
        reg_alpha=0.0,       # another regularizer
        gamma=0.0,           # γ
        objective="reg:squarederror",
        random_state=42,
    )
    xgb_model.fit(X_sk, Y)
    xgb_preds = xgb_model.predict(X_sk)
    xgb_rmse = np.sqrt(mean_squared_error(Y, xgb_preds))

    print(f"\n  XGBoost training RMSE = {xgb_rmse:.2f}")

    print("\n  Training predictions:")
    for i in range(n):
        print(f"    Year={years[i]}  actual={total[i]}  predicted={xgb_preds[i]:.2f}")

    future_xgb = xgb_model.predict(future)
    print("\n  Future (with extrapolation warning):")
    for j in range(2):
        print(f"    2027+j (year={int(future[j][0])}) → {future_xgb[j]:.2f}  "
              f"(⚠ not true extrapolation)")

except ImportError:
    print("\n" + "=" * 68)
    print("PART 3: XGBOOST — SKIPPED (not installed)")
    print("=" * 68)
    print("  To try: pip install xgboost")


# ============================================================================
# PART 4: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Manual (1 tree)       MAE   = {mae:.2f}")
print(f"  sklearn (100 trees)   RMSE  = {rmse_sk:.2f}")
print(f"  Linear Regression     RMSE  = 35.10   (Formula 01)")
print()
print("  HOW XGBOOST WORKS (one sentence):")
print("  → Each new tree is trained on the GRADIENTS of the previous")
print("    prediction — i.e., it fixes the mistakes the previous trees")
print("    made. After 100 trees, the errors are almost gone.")
print()
print("  THE OBJECTIVE FUNCTION (the math above):")
print("  → Obj = LOSS + REGULARIZATION")
print("  → LOSS keeps predictions accurate")
print("  → REGULARIZATION (γ·T + ½·λ·Σw²) keeps trees SIMPLE")
print("  → XGBoost balances both. That is why it does not overfit.")
print()
print("  JIGJIGA NOTE:")
print("  → GBT fits the training data extremely well (RMSE very low)")
print("  → BUT it CANNOT extrapolate to 2027 (year=5 is unseen)")
print("  → For 2027/2028 forecasts, use LINEAR REGRESSION (Formula 04)")
print("  → Use GBT for IN-SAMPLE patterns: which applicants register?")
print()
print("  WUCHALE NOTE:")
print("  → GBT needs history. Train once 2027 data exists.")
print("  → Until then, K-Means (Formula 07) for segmentation.")
print("=" * 68)


# ============================================================================
# REUSE RULES
# ============================================================================
#
#   🔒 NEVER edit:
#      - the objective function math in the comments
#      - w_L*, w_R*, Gain calculations
#      - GradientBoostingRegressor / XGBRegressor construction
#
#   🔁 ALWAYS edit:
#      - years, total
#      - eta, lam, gam (learning rate, L2, leaf penalty)
#      - n_estimators, max_depth
#
#   ⚙️ AUTO-computed:
#      - gradients g, hessians h
#      - G, H per leaf
#      - leaf weights w*
#      - gain per split
#      - predictions
#
#   ⚠️ Trees CANNOT extrapolate — use linear regression for future years
#   ⚠️ XGBoost usually beats sklearn's GBT — install it for production
#   ⚠️ Lower eta needs more trees (eta=0.01 → n_estimators=1000)
#
# ============================================================================
# WHEN TO USE XGBOOST
# ============================================================================
#
#   ✅ USE IT WHEN:
#     - Tabular data (rows and columns)
#     - You want the highest possible accuracy
#     - You have hundreds to millions of rows
#     - Missing values exist (XGBoost handles them natively)
#
#   ❌ DON'T USE IT WHEN:
#     - You need to EXTRAPOLATE to future years (use regression)
#     - You have images/text/audio (use deep learning)
#     - Dataset is < 100 rows (overfit risk)
#
# ============================================================================
