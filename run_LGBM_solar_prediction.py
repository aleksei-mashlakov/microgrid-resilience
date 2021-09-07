
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import hyperopt

# import statsmodels.api as sm
# from scipy.stats import norm
# from sklearn import ensemble
#df_load = pd.read_csv('') ##

df_solar1 = pd.read_csv('./data/ninja_solar_2012.csv', sep=',', skiprows=3)
df_solar1.index = pd.to_datetime(df_solar1.time , format="%Y-%m-%d %H:%M")

df_solar2 = pd.read_csv('./data/ninja_solar_2013.csv', sep=',', skiprows=3)
df_solar2.index = pd.to_datetime(df_solar2.time , format="%Y-%m-%d %H:%M")
df_solar = pd.concat([df_solar1,df_solar2])
df_solar.index = df_solar.index.tz_localize('UTC').tz_convert('Europe/Helsinki').tz_localize(None)
df_solar = df_solar['2013-01-01 00:00:00':'2013-12-31 23:00:00']
index = pd.date_range(start="{}-01-01 00:00:00".format(year), end="{}-12-31 23:00:00".format(year), freq='H')
index = pd.to_datetime(index, format="%Y-%m-%d %H:%M:%S")
df_solar.index = index
df_solar.drop(['time','local_time', 'irradiance_direct', 'irradiance_diffuse', 'temperature'], axis=1, inplace=True)
df_solar.rename(columns={'electricity':'target'}, inplace=True)
df_solar = df_solar*30
# weather
df_weather = pd.read_csv('./data/weather.csv')
df_weather[['Year', 'm', 'd', 'Time']] = df_weather[['Year', 'm', 'd', 'Time']].astype(str)
df_weather.index = df_weather['Year'] + '-' + df_weather['m'] + '-' + df_weather['d'] + ' ' + df_weather['Time']+':00'
df_weather.index = pd.to_datetime(df_weather.index , format="%Y-%m-%d %H:%M:%S")
df_weather.index = df_weather.index.tz_localize('UTC').tz_convert('Europe/Helsinki').tz_localize(None)
df_weather = df_weather['2013-01-01 00:00:00':'2013-12-31 23:00:00']#[['Air temperature (degC)', 'Wind speed (m/s)']]
df_weather = df_weather.fillna(method='bfill').fillna(method='ffill')
df_weather.index = index
from time_features_utils import cyclical_encoding
df_weather = pd.concat([df_weather, cyclical_encoding(df_weather['Wind direction (deg)'], 360) ], axis=1)
df_weather.drop(['Time zone', 'Year', 'm', 'd', 'Time','Wind direction (deg)'], axis=1, inplace=True)
df_solar = pd.concat([create_features(df_solar), df_weather], axis=1)
# holidays
import holidays
holidays = holidays.Finland(years=year).keys()
holidays = [holiday.strftime('%Y-%m-%d') for holiday in holidays]
holiday_mask = df_solar.index.strftime('%Y-%m-%d').isin(pd.to_datetime(holidays).strftime('%Y-%m-%d'))
df_solar['holiday'] = holiday_mask*1
df_solar.head()



# df_scalers = []
df_preds = []
df_covs = []
df_train_preds = []

for data_frame in [df_load, df_solar]:

    target = 'target'
    features = data_frame.columns
    df_scale = data_frame.copy()

#     from sklearn.preprocessing import MinMaxScaler, StandardScaler
#     y_scaler = MinMaxScaler(feature_range=(0, 1))
#     y_scaler.fit_transform(df_scale[[target]])
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     df_scale[df_scale.columns] = scaler.fit_transform(df_scale[df_scale.columns])
#     df_scalers.append(y_scaler)

    preds_list = []
    cov_list = []
    train_preds_list = []

    for month in range(1,13):
        print('Month: {}'.format(month))
        test = df_scale[df_scale.index.month==month]
        train = df_scale[df_scale.index.month!=month]
        train.reset_index(inplace=True)
        train.drop(['index'], axis=1, inplace=True)
        test.reset_index(inplace=True)
        test.drop(['index'], axis=1, inplace=True)

        ytr = train[target].values
        train.drop([target], axis=1, inplace=True)
        xtr = train.values

        yts = test[target].values
        test.drop([target], axis=1, inplace=True)
        xts = test.values


        METHODS = ['Gradient boosting']
        QUANTILES = [0.5]
        quantiles_legend = [str(int(q * 100)) + 'th percentile' for q in QUANTILES]

        x_test = test['sin_hour']
        preds = np.array([(method, q, x)
                          for method in METHODS
                          for q in QUANTILES
                          for x in x_test])
        preds = pd.DataFrame(preds)
        preds.columns = ['method', 'q', 'x']
        preds = preds.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        preds['label'] = np.resize(yts, preds.shape[0])

        preds_full = preds[['method', 'q', 'label']].copy(deep=True)
        x_train_full = train.copy(deep=True)
        x_test_full = test.copy(deep=True)

        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
        # from sklearn.model_selection import cross_val_score
        def acc_model(params):
            from lightgbm import LGBMRegressor
            gbf = LGBMRegressor(seed=0,
                                learning_rate = float(params['learning_rate']),
                                max_depth = int(params['max_depth']),
                                num_leaves = int(params['num_leaves']),
                                num_iterations = int(params['num_iterations'])
                               )
            #print(gbf.get_params())
    #         gbf = ensemble.GradientBoostingRegressor(loss='quantile',
    #                                                  alpha=0.5,
    #                                                  seed=0,
    #                                                  learning_rate = float(params['learning_rate']),
    #                                                  max_depth = int(params['max_depth']),
    #                                                  num_leaves = int(params['num_leaves'])
    #                                                  #max_leaf_nodes = int(params['max_leaf_nodes']),
    #                                                  #min_samples_split = int(params['min_samples_split']),
    #                                                  #min_samples_leaf = int(params['min_samples_leaf']),
    #                                                  #n_estimators = int(params['n_estimators'])
    #                                                 )
            gbf.fit(x_train_full, ytr)
            return quantile_loss(0.5, ytr, gbf.predict(x_train_full)).sum()

        param_space = {
            'num_leaves':hp.quniform('num_leaves', 10, 100, 10),
            'learning_rate' : hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
            'max_depth': hp.quniform('max_depth', 1, 10, 1),
            #'min_samples_split' : hp.quniform('min_samples_split', 2, 10, 1),
            #'min_samples_leaf' : hp.quniform('min_samples_leaf', 1, 10, 1),
            #'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 50, 10),
            'num_iterations': hp.quniform('num_iterations', 100, 1000, 100)
        }

        best = 10000
        def f(params):
            global best
            acc = acc_model(params)
            if acc < best:
                best = acc
                print('new best:', best, params)
            print('new params:', params)
            return {'loss': acc, 'status': STATUS_OK}

        trials = Trials()
        max_evals = 10
        best = fmin(f, param_space,
                    algo=tpe.suggest, #hyperopt.random.suggest
                    max_evals=max_evals,
                    trials=trials)
        print('best:')
        print(best)


        best = {#'learning_rate': best['learning_rate'],
                'learning_rate': best['learning_rate'],
                'max_depth': int(best['max_depth']),
    #             'max_leaf_nodes': int(best['max_leaf_nodes']),
    #             'min_samples_leaf': int(best['min_samples_leaf']),
    #             'min_samples_split': int(best['min_samples_split']),
                'num_iterations': int(best['num_iterations']),
                'num_leaves':int(best['num_leaves'])
        }

        #best = {'learning_rate':0.01, 'n_estimators':1000}
        ## test
        preds_full.loc[preds_full.method == 'Gradient boosting', 'pred'] = \
        np.concatenate([gb_quantile(x_train_full, ytr, x_test_full, q, best)
                        for q in QUANTILES])
        preds_full['quantile_loss'] = quantile_loss(preds_full.q,
                                                    preds_full.label,
                                                    preds_full.pred)
        preds_list.append(preds_full)



        ## train
        x_test = train['sin_hour']
        preds = np.array([(method, q, x)
                          for method in METHODS
                          for q in QUANTILES
                          for x in x_test])
        preds = pd.DataFrame(preds)
        preds.columns = ['method', 'q', 'x']
        preds = preds.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        preds['label'] = np.resize(ytr, preds.shape[0])

        preds_full = preds[['method', 'q', 'label']].copy(deep=True)

        preds_full.loc[preds_full.method == 'Gradient boosting', 'pred'] = \
        np.concatenate([gb_quantile(x_train_full, ytr, x_train_full, q, best)
                        for q in QUANTILES])
        cov = pd.DataFrame((preds_full['label'].values-preds_full['pred'].values).reshape(int(len(preds_full['label'])/24.0),24)).T
        cov_list.append(cov)
        train_preds_list.append(preds_full)

    df_preds.append(preds_list)
    df_covs.append(cov_list)
    df_train_preds.append(train_preds_list)


import pickle
with open(r"load.pickle", "wb") as output_file:
    pickle.dump([df_preds,df_cov df_train_preds], output_file)

preds_full = pd.concat(df_preds[0])
preds_full['time'] = df_solar.index
preds_full = preds_full.set_index('time').drop(['q','method'], axis=1)
preds_full[preds_full['pred']<=0] = 0
preds_full[['label','pred']].plot()

def calculate_mape(y_true, y_pred):
    """ Calculate mean absolute percentage error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_mpe(y_true, y_pred):
    """ Calculate mean percentage error (MPE)"""
    return np.mean((y_true - y_pred) / y_true) * 100

def calculate_mae(y_true, y_pred):
    """ Calculate mean absolute error (MAE)"""
    return np.mean(np.abs(y_true - y_pred)) #* 100

def calculate_rmse(y_true, y_pred):
    """ Calculate root mean square error (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred)**2)) #/ np.sum(y_true)

def print_error_metrics(y_true, y_pred):
    print('MAPE: %f'%calculate_mape(y_true, y_pred))
    print('MPE: %f'%calculate_mpe(y_true, y_pred))
    print('MAE: %f'%calculate_mae(y_true, y_pred))
    print('RMSE: %f'%calculate_rmse(y_true, y_pred))

print_error_metrics(preds_full[preds_full['label']!=0]['label'],preds_full[preds_full['label']!=0]['pred'])
