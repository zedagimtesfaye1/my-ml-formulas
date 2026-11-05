# ============================================================
# FORMULA 02: MEAN SQUARED ERROR (MSE) + RMSE
# PURPOSE: check how good FORMULA 01 predictions are
# ============================================================
#
# ----------------------- MATH -------------------------------
#   MSE  = (1/n) * SUM [ (Y_actual[i] - Y_predicted[i])^2 ]
#   RMSE = SQRT(MSE)
#
# ----------------------- SYMBOLS ----------------------------
#   n              = number of data points
#   Y_actual[i]    = real value from records
#   Y_predicted[i] = model's guess
#   SUM            = add all n items
#   ^2             = square the error (removes negatives, punishes big errors)
#   1/n            = average
#   SQRT           = bring back to real units (students)
#
# ----------------------- TWO METHODS ------------------------
#   METHOD A (manual)  : see the math
#   METHOD B (library) : scikit-learn, faster
#   BOTH give the SAME answer.
#
# ============================================================


# ---------- STEP 1A: FORMULA 01 (manual, from previous file) ----------

def train_manual(X, Y):
    count = len(X)
    average_x = sum(X) / count
    average_y = sum(Y) / count
    top_sum = 0
    bottom_sum = 0
    for i in range(count):
        top_sum = top_sum + (X[i] - average_x) * (Y[i] - average_y)
        bottom_sum = bottom_sum + (X[i] - average_x) * (X[i] - average_x)
    W = top_sum / bottom_sum
    B = average_y - W * average_x
    return W, B


def predict_manual(X, W, B):
    return W * X + B


# ---------- STEP 1B: FORMULA 02 (manual) ----------

def mse_manual(actual, predicted):
    # Math:  MSE = (1/n) * SUM [ (actual[i] - predicted[i])^2 ]
    count = len(actual)                          # n
    total_error = 0                              # SUM = 0
    for i in range(count):
        difference = actual[i] - predicted[i]    # (Y_actual - Y_predicted)
        squared = difference * difference        # ^2
        total_error = total_error + squared      # SUM accumulate
    result = total_error / count                 # (1/n) * SUM
    return result


def rmse_manual(actual, predicted):
    # Math:  RMSE = SQRT(MSE)
    value = mse_manual(actual, predicted)
    return round(value ** 0.5, 1)                # SQRT = **0.5


# ---------- STEP 1C: LIBRARY IMPORTS (never edit) ----------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ---------- STEP 2: YOUR DATA (ONLY EDIT THIS SECTION) ----------
# Same Jigjiga data as Formula 01 - nothing new, nothing random.

years  = [0, 1, 2, 3, 4]
male   = [50, 70, 20, 80, 10]
female = [30, 40, 30, 50, 20]
total  = [80, 110, 50, 130, 30]


# ---------- STEP 3: TRAIN (manual) - just formulas 01 ----------

W_male,   B_male   = train_manual(years, male)
W_female, B_female = train_manual(years, female)
W_total,  B_total  = train_manual(years, total)


# ---------- STEP 4: PREDICT HISTORY (manual) ----------
#
# For MALE (W=-7.0, B=60.0):
#   predict(0)=60.0, predict(1)=53.0, predict(2)=46.0,
#   predict(3)=39.0, predict(4)=32.0
# These become "predicted_male" - all computed, not typed.

predicted_male   = []
predicted_female = []
predicted_total  = []

for year in years:
    predicted_male.append(predict_manual(year, W_male, B_male))
    predicted_female.append(predict_manual(year, W_female, B_female))
    predicted_total.append(predict_manual(year, W_total, B_total))


# ---------- METHOD A: MANUAL MSE + RMSE ----------
#
# For MALE:
#   actual    = [50,  70,  20,  80,  10]
#   predicted = [60,  53,  46,  39,  32]
#   errors    = [-10, 17, -26, 41, -22]
#   squared   = [100, 289, 676, 1681, 484]
#   SUM       = 3230
#   MSE       = 3230 / 5 = 646.0
#   RMSE      = SQRT(646) = 25.4

print("=== METHOD A: MANUAL ===")
print("MSE  MALE   =", mse_manual(male,   predicted_male))
print("MSE  FEMALE =", mse_manual(female, predicted_female))
print("MSE  TOTAL  =", mse_manual(total,  predicted_total))
print("RMSE MALE   =", rmse_manual(male,   predicted_male))
print("RMSE FEMALE =", rmse_manual(female, predicted_female))
print("RMSE TOTAL  =", rmse_manual(total,  predicted_total))


# ---------- METHOD B: LIBRARY (scikit-learn) ----------
#
# Same data. Same result. Library computes faster.

X_2d = []
for year in years:
    X_2d.append([year])

model_male   = LinearRegression().fit(X_2d, male)
model_female = LinearRegression().fit(X_2d, female)
model_total  = LinearRegression().fit(X_2d, total)

lib_pred_male   = model_male.predict(X_2d)      # predicted for all years
lib_pred_female = model_female.predict(X_2d)
lib_pred_total  = model_total.predict(X_2d)

# mean_squared_error does the WHOLE Formula 02 for you
mse_male   = mean_squared_error(male,   lib_pred_male)
mse_female = mean_squared_error(female, lib_pred_female)
mse_total  = mean_squared_error(total,  lib_pred_total)

# RMSE = SQRT(MSE) - library has no RMSE shortcut, so use **0.5
print("\n=== METHOD B: LIBRARY (scikit-learn) ===")
print("MSE  MALE   =", mse_male)
print("MSE  FEMALE =", mse_female)
print("MSE  TOTAL  =", mse_total)
print("RMSE MALE   =", round(mse_male   ** 0.5, 1))
print("RMSE FEMALE =", round(mse_female ** 0.5, 1))
print("RMSE TOTAL  =", round(mse_total  ** 0.5, 1))
