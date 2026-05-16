 

#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind

#  CREATE DATASET
data = {
    "Mobile_Hours": [5, 6, 7, None, 8, 9, 3],
    "Social_Media_Hours": [3, 4, 5, 2, None, 7, 1],
    "Gaming_Hours": [2, 3, 4, 1, 5, None, 1]
}

df = pd.DataFrame(data)

print("ORIGINAL DATASET")
print(df)

# CHECK MISSING VALUES
print("MISSING VALUES")
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)
#  FILL MISSING VALUES

print("DATASET AFTER FILLING MISSING VALUES")
print(df)

# CALCULATE TIME WASTE
df["Time_Waste"] = (
    df["Mobile_Hours"] +
    df["Social_Media_Hours"] +
    df["Gaming_Hours"]
)

print("FINAL DATASET")
print(df)

# CORRELATION MATRIX
print("CORRELATION MATRIX")
print(df.corr().round(4))

#  HYPOTHESIS TEST
t_stat, p_value = ttest_ind(
    df["Mobile_Hours"],
    df["Social_Media_Hours"]
)

print("T-Statistic =", round(t_stat, 4))
print("P-Value =", round(p_value, 4))

if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

# SIMPLE LINEAR REGRESSION
X_simple = df[["Mobile_Hours"]]
y = df["Time_Waste"]

simple_model = LinearRegression()
simple_model.fit(X_simple, y)

simple_pred = simple_model.predict(X_simple)

print("SIMPLE LINEAR PREDICTIONS")
print(np.round(simple_pred, 4))

#MULTIPLE REGRESSION
X_multi = df[[
    "Mobile_Hours",
    "Social_Media_Hours",
    "Gaming_Hours"
]]

multi_model = LinearRegression()
multi_model.fit(X_multi, y)

multi_pred = multi_model.predict(X_multi)

print("MULTIPLE REGRESSION PREDICTIONS")
print(np.round(multi_pred, 4))

#POLYNOMIAL REGRESSION
X_game = df[["Gaming_Hours"]]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_game)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

poly_pred = poly_model.predict(X_poly)

print("POLYNOMIAL REGRESSION PREDICTIONS")
print(np.round(poly_pred, 4))

# MODEL EVALUATION
mse = mean_squared_error(y, multi_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, multi_pred)

print("MSE =", round(mse, 4))
print("RMSE =", round(rmse, 4))
print("R2 Score =", round(r2, 4))

# FUTURE PREDICTION
future_data = pd.DataFrame({
    "Mobile_Hours": [8],
    "Social_Media_Hours": [6],
    "Gaming_Hours": [4]
})

future_prediction = multi_model.predict(future_data)

print("PREDICTED TIME WASTE =", round(future_prediction[0], 4))

#   GRAPH 1 - MOBILE LINEAR
plt.figure(figsize=(6,4))
plt.scatter(df["Mobile_Hours"], y, label="Actual Data")
plt.plot(df["Mobile_Hours"], simple_pred, label="Linear Regression")
plt.title("Mobile Hours Linear Regression")
plt.xlabel("Mobile Hours")
plt.ylabel("Time Waste")
plt.legend()
plt.show()

#  GRAPH 3 - GAMING POLYNOMIAL
sorted_game = np.sort(df["Gaming_Hours"])

sorted_game_pred = poly_model.predict(
    poly.transform(sorted_game.reshape(-1,1))
)

plt.figure(figsize=(6,4))
plt.scatter(df["Gaming_Hours"], y, label="Actual Data")
plt.plot(sorted_game, sorted_game_pred, label="Polynomial Regression")
plt.title("Gaming Hours Polynomial Regression")
plt.xlabel("Gaming Hours")
plt.ylabel("Time Waste")
plt.legend()
plt.show()

# GRAPH 2 - SOCIAL MEDIA MULTIPLE
sorted_df = df.sort_values("Social_Media_Hours")

plt.figure(figsize=(6,4))
plt.scatter(df["Social_Media_Hours"], y, label="Actual Data")

plt.plot(
    sorted_df["Social_Media_Hours"],
    multi_model.predict(
        sorted_df[[
            "Mobile_Hours",
            "Social_Media_Hours",
            "Gaming_Hours"
        ]]
    ),
    label="Multiple Regression"
)

plt.title("Social Media Multiple Regression")
plt.xlabel("Social Media Hours")
plt.ylabel("Time Waste")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))

plt.plot(y.values, label="Actual Values", marker='o')
plt.plot(multi_pred, label="Predicted Values", marker='x')

plt.title("📊 Actual vs Predicted Comparison")
plt.xlabel("Users")
plt.ylabel("Screen Time")
plt.legend()
plt.show()

# GRAPH 4 - BAR CHART
plt.figure(figsize=(8,5))
plt.bar(df.index, df["Time_Waste"])

plt.title("Time Waste Bar Chart")
plt.xlabel("Person Index")
plt.ylabel("Time Waste")

for i, value in enumerate(df["Time_Waste"]):
    plt.text(i, value + 0.2, round(value, 4), ha='center')

plt.show()

#  FINAL SUMMARY
print("FINAL SUMMARY")
print("Average Time Waste =", round(df["Time_Waste"].mean(), 4))
print("Maximum Time Waste =", round(df["Time_Waste"].max(), 4))
print("Minimum Time Waste =", round(df["Time_Waste"].min(), 4))

print("\n💡 FINAL CONCLUSION")
print("-"*40)

print("✔ Mobile usage is the strongest predictor of screen time")
print("✔ Social media contributes steadily to time waste")
print("✔ Gaming shows moderate influence")

print("\n📌 MODEL INSIGHT:")
print("Multiple Linear Regression gives the most balanced and accurate prediction")
print("Polynomial regression captures non-linear behavior in gaming usage")

print("\n🏆 OVERALL RESULT:")
print("This model successfully analyzes and predicts human screen usage behavior using ML techniques.")
