import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


data = pd.read_csv("cardekho_dataset.csv")

num_col = data.select_dtypes("number").columns.drop("selling_price")

obj = data.select_dtypes("object").columns

data = pd.get_dummies(data, columns=obj, drop_first=True)

x = data.drop("selling_price", axis = 1)
y = data["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
scalar = StandardScaler()
x_train[num_col] = scalar.fit_transform(x_train[num_col])
x_test[num_col] = scalar.transform(x_test[num_col])

y_train_scaled = np.log1p(y_train)
model = LinearRegression()

model.fit(x_train, y_train_scaled)

y_pred_scaled = model.predict(x_test)
y_pred = np.expm1(y_pred_scaled)

mse = mean_squared_error(y_pred, y_test)
print("RMSE: ", mse**0.5)