import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from sklearn.metrics import mean_squared_error


diamonds = pd.DataFrame(pd.read_csv('diamonds.csv'))
diamonds.drop_duplicates(inplace=True)
diamonds = diamonds.drop(diamonds[(diamonds.x <2) | (diamonds.z >15) | (diamonds.y >20) | (diamonds.carat >4) | (diamonds.depth < 49) | (diamonds.depth > 75)].index)

encoded_d = ce.OneHotEncoder(cols='cut', handle_unknown='return_nan',return_df=True, use_cat_names=True)
data_encoded = encoded_d.fit_transform(diamonds)
encoded_c = ce.OneHotEncoder(cols='color', handle_unknown='return_nan',return_df=True, use_cat_names=True)
data_encoded_color = encoded_c.fit_transform(data_encoded)
encoded_cla = ce.OneHotEncoder(cols='clarity', handle_unknown='return_nan',return_df=True, use_cat_names=True)
data_encoded_final = encoded_cla.fit_transform(data_encoded_color)

model = LinearRegression()
X = data_encoded_final[['carat', 'cut_Premium', 'cut_Good', 'cut_Very Good', 'cut_Ideal','cut_Fair', 'color_E', 'color_I','color_J', 'color_H', 'color_F','color_G', 'color_D', 'clarity_SI1', 'clarity_VS1', 'clarity_VS2','clarity_SI2', 'clarity_VVS2', 'clarity_VVS1', 'clarity_I1','clarity_IF', 'depth', 'table', 'x', 'y', 'z']]
y = data_encoded_final['price']
model.fit(X,y)
y_pred = model.predict(X)

print('Regurar linear regression is over the limit with an RMSE equal to: ',mean_squared_error(y,y_pred,squared=False))