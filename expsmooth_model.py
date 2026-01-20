import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

class ExpSmoothForecaster:
    def __init__(self):
        self.daily_model = None
        self.daily_test_data = None
        self.best_daily_params = None
        self.monthly_model = None
        self.monthly_test_data = None
        self.best_monthly_params = None

    def initialize_data(self, historical_df):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df.set_index('ds', inplace=True)
        df = df.asfreq('D')
        self.df = df

    def initialize_data_monthly(self, historical_df):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['month'] = df['ds'].dt.to_period('M')
        df = df.groupby('month').agg({'y': 'sum'}).reset_index()
        df['ds'] = df['month'].dt.to_timestamp()
        df.set_index('ds', inplace=True)
        self.df_monthly = df

    def grid_search_exp_smooth_daily(self,
                                     trend_options=['add', None],
                                     seasonal_options=['add', None],
                                     seasonal_periods=[7, 14]):
        df = self.df
        test_size = int(len(df) * 0.1)
        train_data = df.iloc[:-test_size]
        test_data = df.iloc[-test_size:]
        best_accuracy = -float('inf')
        best_model = None
        best_params = None
        total_combinations = len(trend_options) * len(seasonal_options) * len(seasonal_periods)
        current_combination = 0

        print("\nStarting Exponential Smoothing daily grid search...")
        for trend, seasonal, sp in product(trend_options, seasonal_options, seasonal_periods):
            current_combination += 1
            try:
                print(f"\rTesting combination {current_combination}/{total_combinations} - Trend: {trend}, Seasonal: {seasonal}, Period: {sp}", end="")
                model = ExponentialSmoothing(
                    train_data['y'],
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=sp if seasonal else None,
                    initialization_method="estimated"
                ).fit(optimized=True)
                pred = model.forecast(len(test_data))
                mae = mean_absolute_error(test_data['y'], pred)
                accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = {'trend': trend, 'seasonal': seasonal, 'seasonal_periods': sp}
                    print(f"\nNew best parameters found! Accuracy: {best_accuracy:.2f}%")
            except Exception:
                continue
        print("\nGrid search completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        self.daily_model = best_model
        self.daily_test_data = test_data
        self.best_daily_params = best_params

    def grid_search_exp_smooth_monthly(self,
                                       trend_options=['add', None],
                                       seasonal_options=['add', None],
                                       seasonal_periods=[12]):
        df = self.df_monthly
        test_size = max(1, int(len(df) * 0.1))
        train_data = df.iloc[:-test_size]
        test_data = df.iloc[-test_size:]
        best_accuracy = -float('inf')
        best_model = None
        best_params = None
        total_combinations = len(trend_options) * len(seasonal_options) * len(seasonal_periods)
        current_combination = 0

        print("\nStarting Exponential Smoothing monthly grid search...")
        for trend, seasonal, sp in product(trend_options, seasonal_options, seasonal_periods):
            current_combination += 1
            try:
                print(f"\rTesting combination {current_combination}/{total_combinations} - Trend: {trend}, Seasonal: {seasonal}, Period: {sp}", end="")
                model = ExponentialSmoothing(
                    train_data['y'],
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=sp if seasonal else None,
                    initialization_method="estimated"
                ).fit(optimized=True)
                pred = model.forecast(len(test_data))
                mae = mean_absolute_error(test_data['y'], pred)
                accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = {'trend': trend, 'seasonal': seasonal, 'seasonal_periods': sp}
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
        pred = self.daily_model.forecast(len(test_data))
        mae = mean_absolute_error(test_data['y'], pred)
        rmse = mean_squared_error(test_data['y'], pred, squared=False)
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
        pred = self.monthly_model.forecast(len(test_data))
        mae = mean_absolute_error(test_data['y'], pred)
        rmse = mean_squared_error(test_data['y'], pred, squared=False)
        accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
        print("\nMonthly Model Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        return mae, rmse, accuracy

    def forecast_future_daily(self, steps):
        if self.daily_model is None:
            raise Exception("Model not trained.")
        pred = self.daily_model.forecast(steps)
        return pred.reset_index().rename(columns={'ds': 'date', 0: 'forecast'})

    def forecast_future_monthly(self, steps):
        if self.monthly_model is None:
            raise Exception("Model not trained.")
        pred = self.monthly_model.forecast(steps)
        return pred.reset_index().rename(columns={'ds': 'date', 0: 'forecast'})
