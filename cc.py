import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Load data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)

# Sidebar for SVR parameters
st.sidebar.header("SVR Hyperparameters")
C = st.sidebar.slider("C (Regularization)", 1, 1000, 100, 1)
epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
gamma = st.sidebar.selectbox("Gamma", options=['scale', 'auto', 0.01, 0.1, 1.0], index=2)

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

# Train SVR
regressor = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
regressor.fit(X_scaled, y_scaled.ravel())

def predict_salary(level):
    level_scaled = sc_X.transform([[level]])
    salary_scaled = regressor.predict(level_scaled)
    return sc_y.inverse_transform(salary_scaled.reshape(-1, 1))[0][0]

# Main UI
st.title("💼 SVR Salary Predictor (Beast Mode)")
level = st.slider("Select Position Level", float(X.min()), float(X.max()), 6.5, 0.1)
if st.button("Predict Salary"):
    salary = predict_salary(level)
    st.success(f"Predicted Salary for Level {level}: {salary:,.2f}")

# Sidebar branding and instructions
st.sidebar.markdown("## 🦾 SVR Salary App")
st.sidebar.info(
    "Adjust the hyperparameters, select a position level, and click **Predict Salary** to see the result. "
    "You can also view and download predictions for all levels below."
)

# Show predictions for all levels
st.subheader("Predicted Salaries for All Levels")
levels = np.arange(float(X.min()), float(X.max())+0.1, 1)
predicted_salaries = [predict_salary(lvl) for lvl in levels]
results_df = pd.DataFrame({'Level': levels, 'Predicted Salary': predicted_salaries})
st.dataframe(results_df.style.format({"Predicted Salary": "{:,.2f}"}))

# Download predictions
csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Predictions as CSV", csv, "predicted_salaries.csv", "text/csv")

# Enhanced plot
st.subheader("SVR Regression Curve")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color='red', label='Actual', s=60, edgecolor='k')
X_grid = np.arange(min(X), max(X)+0.01, 0.01).reshape(-1, 1)
y_pred_curve = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1))
ax.plot(X_grid, y_pred_curve, color='blue', label='SVR Model', linewidth=2)
ax.scatter(level, predict_salary(level), color='green', label='Your Prediction', s=100, marker='*')
ax.set_xlabel('Position Level')
ax.set_ylabel('Salary')
ax.set_title('SVR Regression Curve')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
st.pyplot(fig)