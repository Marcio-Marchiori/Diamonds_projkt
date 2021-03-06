import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


diamonds = pd.DataFrame(pd.read_csv('diamonds.csv'))
diamonds.drop_duplicates(inplace=True)
cols_use = ['carat', 'depth', 'table', 'x', 'y', 'z']

diamonds = diamonds.drop(diamonds[(diamonds.x <2) | (diamonds.z >15) | (diamonds.z <1) | (diamonds.y >20) | (diamonds.carat >3) | (diamonds.depth < 49) | (diamonds.depth > 75)].index)

diamonds['price'] = np.log10(diamonds['price'])

unique_color = diamonds['color'].unique()

diamonds[cols_use] = scaler.fit_transform(diamonds[cols_use])
#diamonds['carat'] = (diamonds['carat']-diamonds['carat'].mean())/diamonds['carat'].std()


encoded_d = ce.OneHotEncoder(cols='cut', handle_unknown='return_nan',return_df=True, use_cat_names=True)
data_encoded = encoded_d.fit_transform(diamonds[10000:])
encoded_c = ce.OneHotEncoder(cols='clarity', handle_unknown='return_nan',return_df=True, use_cat_names=True)
data_encoded_final = encoded_c.fit_transform(data_encoded)


def reg_t(dados_por_cor):
    model = LinearRegression()
    X = dados_por_cor[['carat', 'cut_Premium', 'cut_Good', 'cut_Very Good', 'cut_Ideal','cut_Fair', 'clarity_SI1', 'clarity_VS1', 'clarity_VS2','clarity_SI2', 'clarity_VVS2', 'clarity_VVS1', 'clarity_I1','clarity_IF', 'depth', 'table', 'x', 'y', 'z']]
    y = dados_por_cor['price']
    return model.fit(X,y)



y_pred = model.predict(X)

print('Regurar linear regression is over the limit with an RMSE equal to: ',mean_squared_error(y,y_pred,squared=False))