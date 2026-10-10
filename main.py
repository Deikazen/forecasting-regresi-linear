import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data dari File Excel/CSV
# Sesuaikan nama file dan sheet name
df = pd.read_excel('studi_kasus_forecasting.xlsx', sheet_name='python')

# 2. Penyiapan Fitur (X = Tahun) dan Target (y = Populasi)
X = df[['Tahun']]  # Independent variable (2D array)
y = df['Populasi']  # Dependent variable

# 3. Training Model Regresi Linear
model = LinearRegression()
model.fit(X, y)

# 4. Prediksi
y_pred = model.predict(X)
df['Prediksi'] = y_pred

# 5. Evaluasi Model (R2 & RMSE)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R-Squared: {r2:.8f}")
print(f"RMSE     : {rmse:,.2f}")

# 6. Visualisasi Grafik
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Tahun')
plt.ylabel('Populasi')
plt.title('Forecasting Populasi dengan Regresi Linear')
plt.legend()
plt.grid(True)
plt.show()
