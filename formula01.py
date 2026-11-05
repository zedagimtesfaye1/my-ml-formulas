# ============================================================
# FORMULA 01: LINEAR REGRESSION  (Y = W * X + B)
# PURPOSE: predict future numbers from past numbers
# ============================================================
#
# ----------------------- MATH -------------------------------
#   Y = W * X + B
#   W = SUM[(X[i]-mean_X)*(Y[i]-mean_Y)] / SUM[(X[i]-mean_X)^2]
#   B = mean_Y - W * mean_X
#
# ----------------------- SYMBOLS ----------------------------
#   X  = input (years)         |  W = slope (change per year)
#   Y  = output (counts)       |  B = starting point (at X=0)
#
# ----------------------- TWO METHODS ------------------------
#   METHOD A (manual)  : teach yourself the math
#   METHOD B (library) : scikit-learn, production-ready
#   BOTH give the SAME answer.
#
# ----------------------- REUSE RULES ------------------------
#   NEVER EDIT:
#     - train_manual() and predict_manual()
#     - library model creation lines
#   ALWAYS EDIT:
#     - years, male, female, total lists (STEP 2)
#
# ============================================================


# ---------- STEP 1A: MANUAL FORMULA (never edit) ----------

def train_manual(X, Y):
    # Math:  W = SUM[(X[i]-mean_X)*(Y[i]-mean_Y)] / SUM[(X[i]-mean_X)^2]
    #        B = mean_Y - W * mean_X
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
    # Math:  Y = W * X + B
    return W * X + B


# ---------- STEP 1B: LIBRARY IMPORT (never edit) ----------

from sklearn.linear_model import LinearRegression


# ---------- STEP 2: YOUR DATA (ONLY EDIT THIS SECTION) ----------
#
# Jigjiga history:
#   years  [0,1,2,3,4]     = 2022, 2023, 2024, 2025, 2026
#   male   [50,70,20,80,10]  = real male counts from your records
#   female [30,40,30,50,20]  = real female counts
#   total  [80,110,50,130,30] = real totals

years  = [0, 1, 2, 3, 4]              # X values
male   = [50, 70, 20, 80, 10]         # Y values (male)
female = [30, 40, 30, 50, 20]         # Y values (female)
total  = [80, 110, 50, 130, 30]       # Y values (total)

year_2027 = 5                         # 2027 = offset 5
year_2028 = 6                         # 2028 = offset 6


# ---------- METHOD A: MANUAL TRAINING ----------
#
# Everything computed by our own train_manual().
# W and B are NOT typed by you - they are RETURNED by the function.

W_male,   B_male   = train_manual(years, male)     # W=-7.0, B=60.0
W_female, B_female = train_manual(years, female)   # W=-1.0, B=36.0
W_total,  B_total  = train_manual(years, total)    # W=-8.0, B=96.0

print("=== METHOD A: MANUAL ===")
print("MALE   W =", W_male,   " B =", B_male)
print("FEMALE W =", W_female, " B =", B_female)
print("TOTAL  W =", W_total,  " B =", B_total)

print("2027 MALE   =", predict_manual(year_2027, W_male,   B_male))
print("2027 FEMALE =", predict_manual(year_2027, W_female, B_female))
print("2027 TOTAL  =", predict_manual(year_2027, W_total,  B_total))
print("2028 MALE   =", predict_manual(year_2028, W_male,   B_male))
print("2028 FEMALE =", predict_manual(year_2028, W_female, B_female))
print("2028 TOTAL  =", predict_manual(year_2028, W_total,  B_total))


# ---------- METHOD B: LIBRARY TRAINING (scikit-learn) ----------
#
# IMPORTANT: scikit-learn needs X as 2D (list of lists), not 1D.
# So [0,1,2,3,4]  becomes  [[0],[1],[2],[3],[4]]
# We build this with a small loop - justified below.

X_2d = []
for year in years:
    X_2d.append([year])          # wrap each number in its own list -> 2D

model_male   = LinearRegression()
model_female = LinearRegression()
model_total  = LinearRegression()

model_male.fit(X_2d, male)       # learns W and B internally
model_female.fit(X_2d, female)
model_total.fit(X_2d, total)

# Extract the SAME W and B from the library:
#   model.coef_[0]      = W  (slope)
#   model.intercept_    = B  (starting point)

print("\n=== METHOD B: LIBRARY (scikit-learn) ===")
print("MALE   W =", model_male.coef_[0],   " B =", model_male.intercept_)
print("FEMALE W =", model_female.coef_[0], " B =", model_female.intercept_)
print("TOTAL  W =", model_total.coef_[0],  " B =", model_total.intercept_)

# Predict 2027 and 2028 (must pass 2D!)
print("2027 MALE   =", model_male.predict([[year_2027]])[0])
print("2027 FEMALE =", model_female.predict([[year_2027]])[0])
print("2027 TOTAL  =", model_total.predict([[year_2027]])[0])
print("2028 MALE   =", model_male.predict([[year_2028]])[0])
print("2028 FEMALE =", model_female.predict([[year_2028]])[0])
print("2028 TOTAL  =", model_total.predict([[year_2028]])[0])
