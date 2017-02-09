import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#Read data
dataframe = pd.read_csv("challenge_dataset.txt")
x_values = dataframe[["x"]]
y_values = dataframe[["y"]]

#Training the model
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
