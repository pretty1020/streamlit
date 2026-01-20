import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

class ProphetForecaster:
    def __init__(self, regressors=None, holidays_df=None):
        self.daily_model = None
        self.daily_test_data = None
        self.regressors = regressors or []
        self.holidays_df = holidays_df
        self.best_daily_params = None
        self.monthly_model = None
        self.monthly_test_data = None
        self.best_monthly_params = None

    def initialize_data(self, historical_df):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df.sort_values('ds', inplace=True)
        self.daily_data = df

    def initialize_data_monthly(self, historical_df):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['month'] = df['ds'].dt.to_period('M')
        agg_dict = {'y': 'sum'}
        for reg in self.regressors:
            agg_dict[reg] = 'sum'
        df = df.groupby('month').agg(agg_dict).reset_index()
        df['ds'] = df['month'].dt.to_timestamp()
        df.sort_values('ds', inplace=True)
        self.monthly_data = df

    def grid_search_daily(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.01, 0.1],
                'seasonality_prior_scale': [0.1, 1],
                'seasonality_mode': ['additive']
            }
        grid = list(product(param_grid['changepoint_prior_scale'],
                            param_grid['seasonality_prior_scale'],
                            param_grid['seasonality_mode']))

        valid_size = max(30, int(len(self.daily_data) * 0.2))
        train_df = self.daily_data.iloc[:-valid_size]
        valid_df = self.daily_data.iloc[-valid_size:]
        best_accuracy = -float('inf')
        best_mae = float('inf')
        best_model = None
        best_params = None
        total_combinations = len(grid)
        current_combination = 0

        print("\nStarting Prophet daily grid search...")
        for cps, sps, sm in grid:
            current_combination += 1
            try:
                print(f"\rTesting combination {current_combination}/{total_combinations} - CPS: {cps}, SPS: {sps}, Mode: {sm}", end="")
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    seasonality_mode=sm,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    holidays=self.holidays_df
                )
                for reg in self.regressors:
                    model.add_regressor(reg)
                model.fit(train_df)
                future = valid_df[['ds'] + self.regressors].copy()
                forecast = model.predict(future)
                y_true = valid_df['y']
                y_pred = forecast['yhat']
                mae = mean_absolute_error(y_true, y_pred)
                accuracy = max(0, min(100, (1 - mae / y_true.mean()) * 100))
                if (accuracy > best_accuracy) or (accuracy == best_accuracy and mae < best_mae):
                    best_accuracy = accuracy
                    best_mae = mae
                    best_model = model
                    best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps, 'seasonality_mode': sm}
                    print(f"\nNew best parameters found! Accuracy: {best_accuracy:.2f}%, MAE: {best_mae:.2f}")
            except Exception:
                continue
        print("\nGrid search completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        print(f"Best MAE: {best_mae:.2f}")
        self.daily_model = best_model
        self.daily_test_data = valid_df
        self.best_daily_params = best_params

    def grid_search_monthly(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.01, 0.1],
                'seasonality_prior_scale': [0.1, 1],
                'seasonality_mode': ['additive']
            }
        grid = list(product(param_grid['changepoint_prior_scale'],
                            param_grid['seasonality_prior_scale'],
                            param_grid['seasonality_mode']))
        valid_size = max(3, int(len(self.monthly_data) * 0.2))
        train_df = self.monthly_data.iloc[:-valid_size]
        valid_df = self.monthly_data.iloc[-valid_size:]
        best_accuracy = -float('inf')
        best_mae = float('inf')
        best_model = None
        best_params = None
        total_combinations = len(grid)
        current_combination = 0

        print("\nStarting Prophet monthly grid search...")
        for cps, sps, sm in grid:
            current_combination += 1
            try:
                print(f"\rTesting combination {current_combination}/{total_combinations} - CPS: {cps}, SPS: {sps}, Mode: {sm}", end="")
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    seasonality_mode=sm,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    holidays=self.holidays_df
                )
                for reg in self.regressors:
                    model.add_regressor(reg)
                model.fit(train_df)
                future = valid_df[['ds'] + self.regressors].copy()
                forecast = model.predict(future)
                y_true = valid_df['y']
                y_pred = forecast['yhat']
                mae = mean_absolute_error(y_true, y_pred)
                accuracy = max(0, min(100, (1 - mae / y_true.mean()) * 100))
                if (accuracy > best_accuracy) or (accuracy == best_accuracy and mae < best_mae):
                    best_accuracy = accuracy
                    best_mae = mae
                    best_model = model
                    best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps, 'seasonality_mode': sm}
                    print(f"\nNew best parameters found! Accuracy: {best_accuracy:.2f}%, MAE: {best_mae:.2f}")
            except Exception:
                continue
        print("\nGrid search completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        print(f"Best MAE: {best_mae:.2f}")
        self.monthly_model = best_model
        self.monthly_test_data = valid_df
        self.best_monthly_params = best_params

    def evaluate_daily(self):
        if self.daily_model is None or self.daily_test_data is None:
            raise Exception("Daily model not trained or no test data.")
        future = self.daily_test_data[['ds'] + self.regressors].copy()
        forecast = self.daily_model.predict(future)
        y_true = self.daily_test_data['y']
        y_pred = forecast['yhat']
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        accuracy = max(0, min(100, (1 - mae / y_true.mean()) * 100))
        print("\nDaily Model Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        return mae, rmse, accuracy

    def evaluate_monthly(self):
        if self.monthly_model is None or self.monthly_test_data is None:
            raise Exception("Monthly model not trained or no test data.")
        future = self.monthly_test_data[['ds'] + self.regressors].copy()
        forecast = self.monthly_model.predict(future)
        y_true = self.monthly_test_data['y']
        y_pred = forecast['yhat']
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        accuracy = max(0, min(100, (1 - mae / y_true.mean()) * 100))
        print("\nMonthly Model Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        return mae, rmse, accuracy

    def forecast_future_daily(self, days=30):
        if self.daily_model is None:
            raise Exception("Daily model not trained.")
        last_date = self.daily_data['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        for reg in self.regressors:
            future_df[reg] = 0
        forecast = self.daily_model.predict(future_df)
        return forecast[['ds', 'yhat']]

    def forecast_future_monthly(self, months=12):
        if self.monthly_model is None:
            raise Exception("Monthly model not trained.")
        last_date = self.monthly_data['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=months, freq='MS')
        future_df = pd.DataFrame({'ds': future_dates})
        for reg in self.regressors:
            future_df[reg] = 0
        forecast = self.monthly_model.predict(future_df)
        return forecast[['ds', 'yhat']]
