import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

class ARIMAForecaster:
    def __init__(self):
        self.daily_model = None
        self.daily_test_data = None
        self.best_daily_params = None
        self.exog_columns = None
        self.monthly_model = None
        self.monthly_test_data = None
        self.best_monthly_params = None

    def initialize_data(self, historical_df, exog_columns=None):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df.set_index('ds', inplace=True)
        df = df.asfreq('D')
        for col in exog_columns or []:
            if col not in df.columns:
                df[col] = 0
        self.df = df
        self.exog_columns = exog_columns or []

    def initialize_data_monthly(self, historical_df, exog_columns=None):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['month'] = df['ds'].dt.to_period('M')
        agg_dict = {'y': 'sum'}
        for col in exog_columns or []:
            agg_dict[col] = 'sum'
        df = df.groupby('month').agg(agg_dict).reset_index()
        df['ds'] = df['month'].dt.to_timestamp()
        df.set_index('ds', inplace=True)
        for col in exog_columns or []:
            if col not in df.columns:
                df[col] = 0
        self.df_monthly = df
        self.exog_columns = exog_columns or []

    def grid_search_arima_daily(self, p=[1,2], d=[0,1], q=[1,2]):
        df = self.df
        test_size = int(len(df) * 0.1)
        train_data = df.iloc[:-test_size]
        test_data = df.iloc[-test_size:]
        best_accuracy = -float('inf')
        best_model = None
        best_params = None
        total_combinations = len(p) * len(d) * len(q)
        current_combination = 0

        print("\nStarting ARIMA daily grid search...")
        for order in product(p, d, q):
            current_combination += 1
            try:
                print(f"\rTesting combination {current_combination}/{total_combinations} - Order: {order}", end="")
                model = ARIMA(
                    endog=train_data['y'],
                    exog=train_data[self.exog_columns] if self.exog_columns else None,
                    order=order
                ).fit()
                pred = model.forecast(steps=len(test_data), exog=test_data[self.exog_columns] if self.exog_columns else None)
                mae = mean_absolute_error(test_data['y'], pred)
                accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = {'order': order}
                    print(f"\nNew best parameters found! Accuracy: {best_accuracy:.2f}%")
            except Exception:
                continue
        print("\nGrid search completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        self.daily_model = best_model
        self.daily_test_data = test_data
        self.best_daily_params = best_params

    def grid_search_arima_monthly(self, p=[1,2], d=[0,1], q=[1,2]):
        df = self.df_monthly
        test_size = max(1, int(len(df) * 0.1))
        train_data = df.iloc[:-test_size]
        test_data = df.iloc[-test_size:]
        best_accuracy = -float('inf')
        best_model = None
        best_params = None
        total_combinations = len(p) * len(d) * len(q)
        current_combination = 0

        print("\nStarting ARIMA monthly grid search...")
        for order in product(p, d, q):
            current_combination += 1
            try:
                print(f"\rTesting combination {current_combination}/{total_combinations} - Order: {order}", end="")
                model = ARIMA(
                    endog=train_data['y'],
                    exog=train_data[self.exog_columns] if self.exog_columns else None,
                    order=order
                ).fit()
                pred = model.forecast(steps=len(test_data), exog=test_data[self.exog_columns] if self.exog_columns else None)
                mae = mean_absolute_error(test_data['y'], pred)
                accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = {'order': order}
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
        pred = self.daily_model.forecast(steps=len(test_data), exog=test_data[self.exog_columns] if self.exog_columns else None)
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
        pred = self.monthly_model.forecast(steps=len(test_data), exog=test_data[self.exog_columns] if self.exog_columns else None)
        mae = mean_absolute_error(test_data['y'], pred)
        rmse = mean_squared_error(test_data['y'], pred, squared=False)
        accuracy = max(0, min(100, (1 - mae / test_data['y'].mean()) * 100))
        print("\nMonthly Model Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        return mae, rmse, accuracy

    def forecast_future_daily(self, steps, exog_future=None):
        if self.daily_model is None:
            raise Exception("Model not trained.")
        pred = self.daily_model.forecast(steps=steps, exog=exog_future)
        # Create future dates
        last_date = self.df.index.max()
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='D')[1:]
        # Convert pred to DataFrame with dates
        if isinstance(pred, pd.Series):
            result = pd.DataFrame({'date': future_dates, 'forecast': pred.values})
        else:
            result = pd.DataFrame({'date': future_dates, 'forecast': pred})
        return result

    def forecast_future_monthly(self, steps, exog_future=None):
        if self.monthly_model is None:
            raise Exception("Model not trained.")
        pred = self.monthly_model.forecast(steps=steps, exog=exog_future)
        # Create future dates
        last_date = self.df_monthly.index.max()
        next_month = last_date + pd.offsets.MonthBegin(1)
        future_dates = pd.date_range(start=next_month, periods=steps, freq='MS')
        # Convert pred to DataFrame with dates
        if isinstance(pred, pd.Series):
            result = pd.DataFrame({'date': future_dates, 'forecast': pred.values})
        else:
            result = pd.DataFrame({'date': future_dates, 'forecast': pred})
        return result
