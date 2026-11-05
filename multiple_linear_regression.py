# ============================================================================
# FORMULA 04: MULTIPLE LINEAR REGRESSION
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga, Ethiopia
# Data   : Student registration 2022-2026
# ============================================================================
#
# ============================================================================
# TEDx SPEECH — READ ALOUD
# ============================================================================
#
# "Simple regression used ONE feature. Multiple regression uses MANY.
#  It is the difference between a telescope with ONE lens — and a
#  telescope with TWENTY lenses working together.
#
#  For Jigjiga, simple regression on year alone predicted total
#  enrollment with an error of 35 students. Multiple regression uses
#  BOTH year AND female enrollment. The error drops to 8.48.
#  FOUR TIMES BETTER. Because it finally sees what is happening.
#
#  This is why AI beats human guessing. Not because it is smarter.
#  Because it can listen to many voices at once."
#
# ============================================================================
# EXACT INTERNATIONAL FORMULA
# ============================================================================
#
#   Ŷ = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ          ← equation form
#
#   Ŷ = Xβ                                       ← matrix form
#
#   β̂ = (XᵀX)⁻¹ · XᵀY                          ← normal equation (solution)
#
# SYMBOLS:
#   Ŷ   = predicted output vector
#   β₀  = intercept (baseline)
#   β₁…βₚ = coefficients (how much each feature moves Y)
#   X   = design matrix (n rows × (p+1) columns, first column = 1s)
#   Y   = actual output vector
#   Xᵀ  = transpose of X (rows ↔ columns)
#   ⁻¹  = matrix inverse
#   p   = number of features (2 here: year, female)
#   n   = number of data points (5 here)
#
# ============================================================================
# WHY IT MATTERS — Your Jigjiga Data, Measured
# ============================================================================
#
#   Model                    Features       RMSE      Meaning
#   -----------------------  -------------  --------  ----------------------
#   Simple Regression (F01)  year only      35.1      off by 35 students
#   Multiple Regression(F04) year + female  8.48      off by 8.5 students
#
#   IMPROVEMENT = 4.1× better.
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA DATA
# ============================================================================
#
#   DATA:
#     Year   (X₁) = [0, 1, 2, 3, 4]
#     Female (X₂) = [30, 40, 30, 50, 20]
#     Total  (Y)  = [80, 110, 50, 130, 30]
#     n = 5, p = 2
#
#   STEP 1 — DESIGN MATRIX X (add column of 1s for β₀):
#
#          β₀  Year  Female
#     X = [[1,   0,    30 ],      Y = [ 80 ]
#          [1,   1,    40 ],          [110 ]
#          [1,   2,    30 ],          [ 50 ]
#          [1,   3,    50 ],          [130 ]
#          [1,   4,    20 ]]          [ 30 ]
#
#   STEP 2 — XᵀX  (3×3)  — each cell is a sum over all 5 rows:
#
#     [0][0]=Σ1·1            = 5
#     [0][1]=Σ1·Year         = 0+1+2+3+4       = 10
#     [0][2]=Σ1·Female       = 30+40+30+50+20   = 170
#     [1][1]=ΣYear·Year      = 0+1+4+9+16       = 30
#     [1][2]=ΣYear·Female    = 0+40+60+150+80   = 330
#     [2][2]=ΣFemale·Female  = 900+1600+900+2500+400 = 6300
#
#     XᵀX = [[   5,   10,   170 ],
#            [  10,   30,   330 ],
#            [ 170,  330,  6300 ]]
#
#   STEP 3 — XᵀY  (3×1):
#
#     [0]=ΣTotal          = 80+110+50+130+30           = 400
#     [1]=ΣYear·Total     = 0+110+100+390+120          = 720
#     [2]=ΣFemale·Total   = 2400+4400+1500+6500+600    = 15400
#
#     XᵀY = [ 400, 720, 15400 ]ᵀ
#
#   STEP 4 — SOLVE  XᵀX · β = XᵀY  →  β = (XᵀX)⁻¹·XᵀY
#
#     5β₀   + 10β₁  + 170β₂  = 400
#     10β₀  + 30β₁  + 330β₂  = 720
#     170β₀ + 330β₁ + 6300β₂ = 15400
#
#     Solution:  β₀ = -25.41
#                β₁ =  -4.63
#                β₂ =   3.37
#
#   STEP 5 — THE LEARNED MODEL:
#
#     Total = -25.41 - 4.63·Year + 3.37·Female
#
#   STEP 6 — VERIFY (plug real data back in):
#
#     Year Female Actual Predicted = -25.41-4.63·Year+3.37·Female  Error
#     0    30     80     75.69                                     -4.31
#     1    40     110    104.76                                    -5.24
#     2    30     50     66.43                                     +16.43
#     3    50     130    129.20                                    -0.80
#     4    20     30     23.47                                     -6.53
#
#     MSE  = (18.58+27.46+269.94+0.64+42.64) / 5 = 71.85
#     RMSE = √71.85 = 8.48
#
#   STEP 7 — PREDICT 2027 and 2028:
#     For future years we estimate Female first using Formula 01 trend
#     (Female W ≈ -1, B ≈ 36):  2027 Female ≈ 31, 2028 Female ≈ 30
#
#     2027 (Year=5, Female=31): -25.41 - 4.63·5 + 3.37·31 = 55.9
#     2028 (Year=6, Female=30): -25.41 - 4.63·6 + 3.37·30 = 47.9
#
# ============================================================================
# HOW TO REUSE
# ============================================================================
#   ONLY edit year, female, total, and the future lists.
#   NEVER edit the matrix math or the sklearn calls.
#   All lists must have the SAME length.
# ============================================================================


# ============================================================================
# PART 1: MANUAL VERSION — numpy (from scratch, transparent math)
# ============================================================================
#
#   numpy is the standard scientific library for matrix math.
#   Install once:  pip install numpy
#
# ============================================================================

import numpy as np

print("=" * 68)
print("FORMULA 04: MULTIPLE LINEAR REGRESSION")
print("=" * 68)

# ---------- STEP 1: YOUR DATA (ONLY EDIT THIS SECTION) ----------

year   = [0, 1, 2, 3, 4]            # X₁ : 2022→0 ... 2026→4
female = [30, 40, 30, 50, 20]       # X₂ : real female counts
total  = [80, 110, 50, 130, 30]     # Y  : real total counts

future_year   = [5, 6]              # 2027, 2028
future_female = [31, 30]            # estimated from Formula 01 trend

# ---------- STEP 2: BUILD DESIGN MATRIX X (NEVER EDIT) ----------
# Column of 1s multiplies β₀ (intercept)

X = []
for i in range(len(year)):
    X.append([1, year[i], female[i]])   # [1, X₁, X₂]
X = np.array(X)                         # shape: (5, 3)

Y = np.array(total)                     # shape: (5,)

# ---------- STEP 3: COMPUTE XᵀX and XᵀY (NEVER EDIT) ----------
# Math:  XᵀX = X.T @ X        (3×3 matrix)
#        XᵀY = X.T @ Y        (3×1 vector)
# '@' in numpy means matrix multiply.

XtX = X.T @ X
XtY = X.T @ Y

print("\n--- MANUAL VERSION (numpy) ---")
print("\nSTEP 2: Design matrix X:")
print(X)
print("\nSTEP 3: XᵀX (3×3):")
print(XtX)
print("\nSTEP 3: XᵀY (3×1):")
print(XtY)

# ---------- STEP 4: SOLVE β = (XᵀX)⁻¹·XᵀY (NEVER EDIT) ----------
# np.linalg.solve(XtX, XtY) solves  XtX · β = XtY
# It is the same as (XᵀX)⁻¹ · XᵀY but numerically safer.

beta = np.linalg.solve(XtX, XtY)        # [β₀, β₁, β₂]

b0, b1, b2 = beta[0], beta[1], beta[2]

print("\nSTEP 4: β (coefficients) = [β₀, β₁, β₂]")
print(beta)
print(f"\nSTEP 5: Model = Total = {b0:.2f} + {b1:.2f}·Year + {b2:.2f}·Female")

# ---------- STEP 5: VERIFY WITH YOUR DATA (NEVER EDIT) ----------

predicted = X @ beta                    # matrix multiply: X · β
errors    = Y - predicted
mse       = np.mean(errors ** 2)
rmse      = np.sqrt(mse)

print("\nSTEP 6: Verify on your data")
for i in range(len(year)):
    print(f"  Year={year[i]}  Female={female[i]:>2}  "
          f"Actual={total[i]:>3}  Predicted={predicted[i]:7.2f}  "
          f"Error={errors[i]:+7.2f}")

print(f"\n  MSE  = {mse:.2f}")
print(f"  RMSE = {rmse:.2f}   (simple regression was 35.1 → 4× better)")

# ---------- STEP 6: PREDICT FUTURE YEARS (NEVER EDIT) ----------

print("\nSTEP 7: Predictions")
for i in range(len(future_year)):
    yhat = b0 + b1 * future_year[i] + b2 * future_female[i]
    print(f"  Year offset {future_year[i]} "
          f"(Female={future_female[i]}) → predicted total = {yhat:.1f}")


# ============================================================================
# PART 2: LIBRARY VERSION — scikit-learn (one line, production code)
# ============================================================================
#
#   Install once:  pip install scikit-learn
#   Same formula  β̂ = (XᵀX)⁻¹·XᵀY  runs inside.
#   sklearn adds the 1s column AUTOMATICALLY — do not include it.
#
# ============================================================================

from sklearn.linear_model import LinearRegression

print("\n" + "=" * 68)
print("LIBRARY VERSION (scikit-learn LinearRegression)")
print("=" * 68)

# ---------- BUILD 2D FEATURE MATRIX (NEVER EDIT) ----------
# sklearn wants X as 2D: [[year, female], [year, female], ...]
# Do NOT add the 1s column — sklearn handles the intercept.

X_lib = []
for i in range(len(year)):
    X_lib.append([year[i], female[i]])

Y_lib = total

# ---------- TRAIN (NEVER EDIT) ----------

model = LinearRegression()
model.fit(X_lib, Y_lib)

# ---------- EXTRACT β₀, β₁, β₂ ----------
#   model.intercept_  = β₀
#   model.coef_[0]    = β₁  (year)
#   model.coef_[1]    = β₂  (female)

lb0 = model.intercept_
lb1 = model.coef_[0]
lb2 = model.coef_[1]

print(f"\nModel = Total = {lb0:.2f} + {lb1:.2f}·Year + {lb2:.2f}·Female")

# ---------- VERIFY (NEVER EDIT) ----------

lib_pred = model.predict(X_lib)
lib_mse  = np.mean((np.array(Y_lib) - lib_pred) ** 2)
lib_rmse = np.sqrt(lib_mse)

print(f"RMSE = {lib_rmse:.2f}")

# ---------- PREDICT FUTURE ----------

print("\nPredictions:")
for i in range(len(future_year)):
    pair = [[future_year[i], future_female[i]]]
    yhat = model.predict(pair)[0]
    print(f"  Year offset {future_year[i]} "
          f"(Female={future_female[i]}) → predicted total = {yhat:.1f}")


# ============================================================================
# PART 3: FINAL SUMMARY — all methods agree
# ============================================================================

print("\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"  Manual (numpy)      → RMSE = {rmse:.2f}")
print(f"  Library (sklearn)   → RMSE = {lib_rmse:.2f}")
print(f"  Simple regression   → RMSE = 35.10   (from Formula 01)")
print(f"  Improvement         → {35.10/lib_rmse:.1f}× better")
print()
print("  2027 prediction : 55.9 students  (range 47.4 – 64.4)")
print("  2028 prediction : 47.9 students  (range 39.4 – 56.4)")
print()
print("  MEANING:")
print("  Every year, we lose 4.63 students (after accounting for females).")
print("  Each extra female adds 3.37 to total enrollment.")
print("  The problem is REAL — not just demographics.")
print()
print("  DECISION:")
print("  Prepare 50 seats with confidence. Plan for the lower bound 47.")
print("  Focus recruitment on females — each one brings ~3.4 total.")
print("=" * 68)


# ============================================================================
# REUSE RULES  (print this every time you open the file)
# ============================================================================
#
#   🔒 NEVER edit:
#      - the design matrix loop (X.append([1, year[i], female[i]]))
#      - XtX = X.T @ X   and   XtY = X.T @ Y
#      - np.linalg.solve(XtX, XtY)
#      - model.fit(X_lib, Y_lib)
#
#   🔁 ALWAYS edit:
#      - year, female, total lists (add new years)
#      - future_year, future_female lists (new targets)
#      - All 3 data lists MUST have the same length
#
#   ⚙️ AUTO-computed (do not type):
#      - β₀, β₁, β₂, MSE, RMSE, predictions
#
# ============================================================================
# WUCHALE — waits for Formula 08 (K-Means, unsupervised)
# ============================================================================
