import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

class SARIMAXForecaster:
    def __init__(self):
        self.daily_model = None
        self.daily_test_data = None
        self.best_daily_params = None
        self.exog_columns = None
        self.monthly_model = None
        self.monthly_test_data = None
        self.best_monthly_params = None

    def initialize_data(self, historical_df, exog_columns):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df.set_index('ds', inplace=True)
        df = df.asfreq('D')
        for col in exog_columns:
            if col not in df.columns:
                df[col] = 0
        self.df = df
        self.exog_columns = exog_columns

    def initialize_data_monthly(self, historical_df, exog_columns):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['month'] = df['ds'].dt.to_period('M')
        agg_dict = {'y': 'sum'}
        for col in exog_columns:
            agg_dict[col] = 'sum'
        df = df.groupby('month').agg(agg_dict).reset_index()
        df['ds'] = df['month'].dt.to_timestamp()
        df.set_index('ds', inplace=True)
        for col in exog_columns:
            if col not in df.columns:
                df[col] = 0
        self.df_monthly = df
        self.exog_columns = exog_columns

    def grid_search_sarimax_daily(self, p=[1,2], d=[1], q=[1,2], P=[1], D=[1], Q=[1], s_list=[14,30]):
        df = self.df
        test_size = int(len(df) * 0.1)
        train_data = df.iloc[:-test_size]
        test_data = df.iloc[-test_size:]
        best_accuracy = -np.inf
        best_model = None
        best_params = None
        total_combinations = len(p) * len(d) * len(q) * len(P) * len(D) * len(Q) * len(s_list)
        current_combination = 0

        print("\nStarting SARIMAX daily grid search...")
        for s in s_list:
            for order in product(p, d, q):
                for seasonal_order in product(P, D, Q):
                    current_combination += 1
                    seasonal_order_full = tuple(list(seasonal_order) + [s])
                    try:
                        print(f"\rTesting combination {current_combination}/{total_combinations} - Order: {order}, Seasonal Order: {seasonal_order_full}", end="")
                        model = SARIMAX(
                            endog=train_data['y'],
                            exog=train_data[self.exog_columns],
                            order=order,
                            seasonal_order=seasonal_order_full,
                            freq='D'
                        ).fit(disp=False)
                        pred = model.get_forecast(steps=len(test_data), exog=test_data[self.exog_columns]).predicted_mean
                        mae = mean_absolute_error(test_data['y'], pred)
                        accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model
                            best_params = {'order': order, 'seasonal_order': seasonal_order_full}
                            print(f"\nNew best parameters found! Accuracy: {best_accuracy:.2f}%")
                    except Exception:
                        continue
        print("\nGrid search completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        self.daily_model = best_model
        self.daily_test_data = test_data
        self.best_daily_params = best_params

    def grid_search_sarimax_monthly(self, p=[1,2], d=[1], q=[1,2], P=[1], D=[1], Q=[1], s_list=[12]):
        df = self.df_monthly
        test_size = max(1, int(len(df) * 0.1))
        train_data = df.iloc[:-test_size]
        test_data = df.iloc[-test_size:]
        best_accuracy = -np.inf
        best_model = None
        best_params = None
        total_combinations = len(p) * len(d) * len(q) * len(P) * len(D) * len(Q) * len(s_list)
        current_combination = 0

        print("\nStarting SARIMAX monthly grid search...")
        for s in s_list:
            for order in product(p, d, q):
                for seasonal_order in product(P, D, Q):
                    current_combination += 1
                    seasonal_order_full = tuple(list(seasonal_order) + [s])
                    try:
                        print(f"\rTesting combination {current_combination}/{total_combinations} - Order: {order}, Seasonal Order: {seasonal_order_full}", end="")
                        model = SARIMAX(
                            endog=train_data['y'],
                            exog=train_data[self.exog_columns],
                            order=order,
                            seasonal_order=seasonal_order_full,
                            freq='MS'
                        ).fit(disp=False)
                        pred = model.get_forecast(steps=len(test_data), exog=test_data[self.exog_columns]).predicted_mean
                        mae = mean_absolute_error(test_data['y'], pred)
                        accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model
                            best_params = {'order': order, 'seasonal_order': seasonal_order_full}
                            print(f"\nNew best parameters found! Accuracy: {best_accuracy:.2f}%")
                    except Exception:
                        continue
        print("\nGrid search completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        self.monthly_model = best_model
        self.monthly_test_data = test_data
        self.best_monthly_params = best_params

    def evaluate_daily(self):
        test_data = self.daily_test_data
        if self.daily_model is None or test_data is None:
            raise Exception("Model not trained or test data not available.")
        pred = self.daily_model.get_forecast(steps=len(test_data), exog=test_data[self.exog_columns]).predicted_mean
        mae = mean_absolute_error(test_data['y'], pred)
        rmse = np.sqrt(mean_squared_error(test_data['y'], pred))
        accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
        print("\nDaily Model Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        return mae, rmse, accuracy

    def evaluate_monthly(self):
        test_data = self.monthly_test_data
        if self.monthly_model is None or test_data is None:
            raise Exception("Model not trained or test data not available.")
        pred = self.monthly_model.get_forecast(steps=len(test_data), exog=test_data[self.exog_columns]).predicted_mean
        mae = mean_absolute_error(test_data['y'], pred)
        rmse = np.sqrt(mean_squared_error(test_data['y'], pred))
        accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
        print("\nMonthly Model Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        return mae, rmse, accuracy

    def forecast_future_daily(self, steps, exog_future=None):
        if self.daily_model is None:
            raise Exception("Model not trained.")
        pred = self.daily_model.get_forecast(steps=steps, exog=exog_future).predicted_mean
        return pred.reset_index().rename(columns={'ds': 'date', 0: 'forecast'})

    def forecast_future_monthly(self, steps, exog_future=None):
        if self.monthly_model is None:
            raise Exception("Model not trained.")
        pred = self.monthly_model.get_forecast(steps=steps, exog=exog_future).predicted_mean
        return pred.reset_index().rename(columns={'ds': 'date', 0: 'forecast'})
