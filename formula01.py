# ============================================================
# FORMULA 01: LINEAR REGRESSION  (Y = W * X + B)
# PURPOSE: predict future numbers from past numbers
# ============================================================


# ---------- STEP 1: THE FORMULA (NEVER EDIT THIS) ----------

def train(X, Y):
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


def predict(X, W, B):
    Y = W * X + B
    return Y


# ---------- STEP 2: YOUR DATA (ONLY EDIT THIS SECTION) ----------

years  = [0, 1, 2, 3, 4]
male   = [50, 70, 20, 80, 10]
female = [30, 40, 30, 50, 20]
total  = [80, 110, 50, 130, 30]

year_2027 = 5
year_2028 = 6


# ---------- STEP 3: RUN THE FORMULA (NEVER EDIT THIS) ----------

W_male,   B_male   = train(years, male)
W_female, B_female = train(years, female)
W_total,  B_total  = train(years, total)

print("MALE   W =", W_male,   " B =", B_male)
print("FEMALE W =", W_female, " B =", B_female)
print("TOTAL  W =", W_total,  " B =", B_total)

print("2027 MALE   =", predict(year_2027, W_male,   B_male))
print("2027 FEMALE =", predict(year_2027, W_female, B_female))
print("2027 TOTAL  =", predict(year_2027, W_total,  B_total))
print("2028 MALE   =", predict(year_2028, W_male,   B_male))
print("2028 FEMALE =", predict(year_2028, W_female, B_female))
print("2028 TOTAL  =", predict(year_2028, W_total,  B_total))
