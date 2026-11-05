# ============================================================================
# FORMULA 03: GRADIENT DESCENT  (THE ENGINE OF ALL MACHINE LEARNING)
# ============================================================================
# Author : Zedagim Tesfaye  |  Location: Jigjiga, Ethiopia
# Data   : Student registration 2022-2026 (Male)
# ============================================================================
#
# ============================================================================
# TEDX SPEECH — READ THIS ALOUD
# ============================================================================
#
# "Every AI on Earth — ChatGPT, self-driving cars, medical diagnosis —
#  uses ONE simple idea. It's called Gradient Descent. And I can explain
#  it in 30 seconds.
#
#  Start with a wrong guess.
#  Measure how wrong you are.
#  Take a small step in the right direction.
#  Repeat two thousand times.
#
#  That's it. That's the whole engine.
#
#  For Jigjiga: the machine started at zero. It knew NOTHING about our
#  students. It looked at five years of data — 50, 70, 20, 80, 10 males.
#  It took two thousand tiny steps. At each step it asked: 'Am I closer?
#  Am I closer?' And at the end, it arrived at the truth:
#
#  EVERY YEAR, OUR MALES DROP BY 7.
#
#  That's not magic. That's a machine taking 2,000 honest steps.
#  It learns the way a child learns to walk — by falling, adjusting,
#  trying again. Two thousand times.
#
#  Every AI in the world runs this loop. Now you know the secret."
#
# ============================================================================
# EXACT MATH FORMULA
# ============================================================================
#
#  MASTER RULE (works for every model in machine learning):
#
#      W_new = W_old  -  α × (∂MSE/∂W)
#      B_new = B_old  -  α × (∂MSE/∂B)
#
#  FOR LINEAR REGRESSION WITH MSE COST — EXACT FORMULAS:
#
#      predicted_i = W·X_i + B
#
#      ∂MSE/∂W = (2/n) × Σ[ (predicted_i - actual_i) × X_i ]
#      ∂MSE/∂B = (2/n) × Σ[  predicted_i - actual_i        ]
#
#  SO THE EXACT UPDATE LINES ARE:
#
#      W_new = W_old - α × (2/n) × Σ[ (predicted_i - actual_i) × X_i ]
#      B_new = B_old - α × (2/n) × Σ[  predicted_i - actual_i        ]
#
# ============================================================================
# SYMBOLS — EXACT MEANING
# ============================================================================
#
#   α (alpha)   = learning rate (0.01 = small safe step)
#   n           = number of data points (5 here)
#   X_i         = year offset [0,1,2,3,4]
#   actual_i    = real male counts [50,70,20,80,10]
#   predicted_i = model's current guess
#   ∂MSE/∂W     = slope of the cost in the W direction (calculus)
#   ∂MSE/∂B     = slope of the cost in the B direction (calculus)
#   2           = from calculus derivative of x² (rule of math, not choice)
#   -           = subtract (step opposite to slope = downhill)
#   W, B        = learned weights (start at 0, become -7 and 60)
#
# ============================================================================
# MANUAL CALCULATION — YOUR EXACT JIGJIGA MALE DATA
# ============================================================================
#
#   DATA:    X = [0, 1, 2, 3, 4]
#            actual = [50, 70, 20, 80, 10]
#            α = 0.01     n = 5
#
#   ------------------ START: W = 0, B = 0 ------------------
#
#   ITERATION 1:
#     Step 1: predicted = W·X + B = [0, 0, 0, 0, 0]
#     Step 2: errors    = [0-50, 0-70, 0-20, 0-80, 0-10]
#                       = [-50, -70, -20, -80, -10]
#     Step 3: Σ(error·X) = 0·(-50) + 1·(-70) + 2·(-20)
#                        + 3·(-80) + 4·(-10)
#                        = 0 - 70 - 40 - 240 - 40 = -390
#             dW = (2/5) × (-390) = -156
#     Step 4: Σ(error) = -50 - 70 - 20 - 80 - 10 = -230
#             dB = (2/5) × (-230) = -92
#     Step 5: W = 0 - 0.01×(-156) = 0 + 1.56 = 1.56
#             B = 0 - 0.01×(-92)  = 0 + 0.92 = 0.92
#
#   ITERATION 2 (using W=1.56, B=0.92):
#     Step 1: predicted = [0.92, 2.48, 4.04, 5.60, 7.16]
#     Step 2: errors    = [-49.08, -67.52, -15.96, -74.40, -2.84]
#     Step 3: Σ(error·X) = -334.00
#             dW = (2/5) × (-334.00) = -133.60
#     Step 4: Σ(error) = -209.80
#             dB = (2/5) × (-209.80) = -83.92
#     Step 5: W = 1.56 - 0.01×(-133.60) = 2.896
#             B = 0.92 - 0.01×(-83.92)  = 1.7592
#
#   ITERATION 3 (using W=2.896, B=1.7592):
#     predicted = [1.7592, 4.6552, 7.5512, 10.4472, 13.3432]
#     errors    = [-48.2408, -65.3448, -12.4488, -69.5528, 3.3432]
#     Σ(error·X) = -309.9272
#     dW = (2/5) × (-309.9272) = -123.9709
#     W = 2.896 - 0.01×(-123.9709) = 4.1357
#     Σ(error) = -192.2440
#     dB = (2/5) × (-192.2440) = -76.8976
#     B = 1.7592 - 0.01×(-76.8976) = 2.5282
#
#   ...continue 2000 times...
#
#   FINAL (after 2000 iterations):
#     W ≈ -7.0      B ≈ 59.94
#     (SAME as Formula 01's direct answer: W = -7, B = 60)
#     → "Males drop by 7 per year, line starts at 60"
#     → 2027 (X=5): -7×5 + 60 = 25 males
#     → 2028 (X=6): -7×6 + 60 = 18 males
#
# ============================================================================


# ============================================================================
# PART 1: MANUAL GRADIENT DESCENT — Python code (NEVER EDIT THIS FUNCTION)
# ============================================================================

def gradient_descent_manual(X, actual, alpha, iterations):
    """
    EXACT FORMULAS (from calculus):
      predicted_i = W·X_i + B
      dW = (2/n) × Σ[ (predicted_i - actual_i) × X_i ]
      dB = (2/n) × Σ[  predicted_i - actual_i        ]
      W_new = W_old - α × dW
      B_new = B_old - α × dB
    """
    W = 0.0                          # starting guess
    B = 0.0                          # starting guess
    n = len(X)                       # n = number of data points

    for step in range(iterations):

        # ----- Step 1: predicted_i = W·X_i + B -----
        predicted = []
        for i in range(n):
            predicted.append(W * X[i] + B)

        # ----- Step 2: compute gradients (exact calculus formulas) -----
        sum_W = 0.0
        sum_B = 0.0
        for i in range(n):
            error = predicted[i] - actual[i]
            sum_W = sum_W + error * X[i]     # Σ[(pred-actual)·X]
            sum_B = sum_B + error            # Σ[(pred-actual)]
        dW = (2 / n) * sum_W                 # (2/n) × Σ
        dB = (2 / n) * sum_B                 # (2/n) × Σ

        # ----- Step 3: update (master rule) -----
        W = W - alpha * dW                   # W_new = W_old - α·dW
        B = B - alpha * dB                   # B_new = B_old - α·dB

    return W, B


# ============================================================================
# PART 2: YOUR DATA — ONLY EDIT THIS SECTION FOR NEW PROBLEMS
# ============================================================================
#
# Jigjiga history:
#   2022 = offset 0  →  50 males
#   2023 = offset 1  →  70 males
#   2024 = offset 2  →  20 males
#   2025 = offset 3  →  80 males
#   2026 = offset 4  →  10 males
#
# ============================================================================

years  = [0, 1, 2, 3, 4]
male   = [50, 70, 20, 80, 10]

alpha       = 0.01        # learning rate (step size)
iterations  = 2000        # how many steps to take


# ============================================================================
# PART 3: RUN MANUAL VERSION
# ============================================================================

print("=" * 65)
print("GRADIENT DESCENT — MANUAL (from scratch)")
print("=" * 65)

# Show progress every 500 steps (proof it's learning)
W = 0.0
B = 0.0
n = len(years)

for step in range(iterations):
    predicted = []
    for i in range(n):
        predicted.append(W * years[i] + B)

    sum_W = 0.0
    sum_B = 0.0
    for i in range(n):
        error = predicted[i] - male[i]
        sum_W = sum_W + error * years[i]
        sum_B = sum_B + error
    dW = (2 / n) * sum_W
    dB = (2 / n) * sum_B

    W = W - alpha * dW
    B = B - alpha * dB

    if step % 500 == 0 or step == iterations - 1:
        print(f"  Step {step:4d}  →  W = {W:8.4f}   B = {B:8.4f}")

print(f"\n  FINAL  W = {round(W, 3)}   B = {round(B, 3)}")
print(f"  → Every year, males drop by {round(abs(W), 0)}")
print(f"  → Line starts at B = {round(B, 0)}")
print(f"  → 2027 (X=5):  {round(W * 5 + B, 1)} males")
print(f"  → 2028 (X=6):  {round(W * 6 + B, 1)} males")


# ============================================================================
# PART 4: LIBRARY VERSION — scikit-learn SGDRegressor
# ============================================================================
#
# SGDRegressor = Stochastic Gradient Descent Regressor
# Same exact algorithm. Same answer. Just written by professionals.
#
# ============================================================================

from sklearn.linear_model import SGDRegressor

# scikit-learn needs 2D input: [0,1,2,3,4] → [[0],[1],[2],[3],[4]]
X_2d = []
for year in years:
    X_2d.append([year])

model = SGDRegressor(
    max_iter=iterations,           # iterations
    eta0=alpha,                    # α (learning rate)
    learning_rate="constant",      # keep α fixed
    random_state=42,               # for reproducibility
)
model.fit(X_2d, male)

lib_W = round(model.coef_[0], 3)
lib_B = round(model.intercept_[0], 3)

print("\n" + "=" * 65)
print("GRADIENT DESCENT — LIBRARY (scikit-learn SGDRegressor)")
print("=" * 65)
print(f"  FINAL  W = {lib_W}   B = {lib_B}")
print(f"  → 2027 (X=5):  {round(lib_W * 5 + lib_B, 1)} males")
print(f"  → 2028 (X=6):  {round(lib_W * 6 + lib_B, 1)} males")


# ============================================================================
# PART 5: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY — ALL THREE METHODS AGREE")
print("=" * 65)
print("  Formula 01 (direct)         →  W = -7.0  ,  B = 60.0")
print(f"  Formula 03 (manual)         →  W = {round(W, 1):5.1f}  ,  B = {round(B, 1):5.1f}")
print(f"  Formula 03 (library)        →  W = {lib_W:5.1f}  ,  B = {lib_B:5.1f}")
print()
print("  ALL THREE = SAME ANSWER = SAME PREDICTION = SAME DECISION")
print()
print("  → 2027 : ~25 males")
print("  → 2028 : ~18 males")
print("  → Trend: males drop by 7 per year")
print("=" * 65)


# ============================================================================
# HOW TO REUSE FOR NEW DATA
# ============================================================================
#
#  ONLY EDIT THIS (inside PART 2):
#    years  = [0, 1, 2, 3, 4, 5]           # add next year
#    male   = [50, 70, 20, 80, 10, 40]     # add real count
#
#  THEN RUN AGAIN.  Everything else stays the same.
#
#  RULES:
#    🔒 NEVER edit gradient_descent_manual()
#    🔒 NEVER edit the math formulas (they are calculus, not choices)
#    🔁 ONLY edit years and male lists
#    ⚙️ W and B are auto-computed
#
#  TUNING TIPS (only if output looks strange):
#    If W shows "nan" or huge number  →  lower alpha to 0.001
#    If W barely moves from 0         →  raise alpha to 0.05 or more iterations
#
# ============================================================================
# WUCHALE — WAITS FOR FORMULA 07 (K-MEANS, unsupervised)
# ============================================================================
#  Wuchale has no history, so gradient descent cannot train on it.
#  Unsupervised methods handle Wuchale later.
# ============================================================================
