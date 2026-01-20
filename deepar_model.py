import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product

import torch
from pytorch_lightning import Trainer
from pytorch_forecasting import TimeSeriesDataSet, DeepAR, PredictionDataLoader

class DeepARForecaster:
    def __init__(self):
        self.daily_model = None
        self.monthly_model = None
        self.daily_test_data = None
        self.monthly_test_data = None
        self.best_daily_params = None
        self.best_monthly_params = None
        self.regressors = None
        self.daily_df = None
        self.monthly_df = None

    def initialize_data(self, historical_df, regressors=None, group_col="series"):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        df["series"] = 0  # single time series
        for reg in regressors or []:
            if reg not in df.columns:
                df[reg] = 0
        self.daily_df = df
        self.regressors = regressors or []

    def initialize_data_monthly(self, historical_df, regressors=None, group_col="series"):
        df = historical_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df["series"] = 0
        agg_dict = {"y": "sum"}
        for reg in regressors or []:
            agg_dict[reg] = "sum"
        df['month'] = df['ds'].dt.to_period('M')
        df = df.groupby(['series', 'month']).agg(agg_dict).reset_index()
        df['ds'] = df['month'].dt.to_timestamp()
        for reg in regressors or []:
            if reg not in df.columns:
                df[reg] = 0
        self.monthly_df = df
        self.regressors = regressors or []

    def _make_tsdataset(self, df, max_encoder_length, max_prediction_length):
        return TimeSeriesDataSet(
            df,
            time_idx="ds",
            target="y",
            group_ids=["series"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["series"],
            time_varying_unknown_reals=["y"] + self.regressors,
            time_varying_known_reals=["ds"] + self.regressors,
        )

    def grid_search_daily(self, max_encoder_length=30, max_prediction_length=7, param_grid=None):
        df = self.daily_df.copy()
        df["ds"] = (df["ds"] - df["ds"].min()).dt.days
        test_size = max_prediction_length
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-(max_encoder_length + test_size):]  # ensure enough for context + prediction

        if param_grid is None:
            param_grid = {
                "hidden_size": [16, 32],
                "dropout": [0.1, 0.3]
            }

        best_mae = np.inf
        best_model = None
        best_params = None
        for h, dr in product(param_grid["hidden_size"], param_grid["dropout"]):
            try:
                train_ds = self._make_tsdataset(train_df, max_encoder_length, max_prediction_length)
                val_ds = self._make_tsdataset(test_df, max_encoder_length, max_prediction_length)
                train_dl = train_ds.to_dataloader(train=True, batch_size=32, num_workers=0)
                val_dl = val_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

                model = DeepAR.from_dataset(
                    train_ds,
                    hidden_size=h,
                    dropout=dr,
                    learning_rate=1e-2
                )
                trainer = Trainer(max_epochs=5, logger=False, enable_checkpointing=False, enable_progress_bar=False)
                trainer.fit(model, train_dl)
                preds = model.predict(val_dl).numpy().flatten()
                true = test_df.tail(max_prediction_length)["y"].values
                mae = mean_absolute_error(true, preds)
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_params = {"hidden_size": h, "dropout": dr}
            except Exception as e:
                continue
        self.daily_model = best_model
        self.daily_test_data = test_df.tail(max_prediction_length)
        self.best_daily_params = best_params

    def grid_search_monthly(self, max_encoder_length=6, max_prediction_length=3, param_grid=None):
        df = self.monthly_df.copy()
        df["ds"] = (df["ds"] - df["ds"].min()).dt.days
        test_size = max_prediction_length
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-(max_encoder_length + test_size):]

        if param_grid is None:
            param_grid = {
                "hidden_size": [16, 32],
                "dropout": [0.1, 0.3]
            }

        best_mae = np.inf
        best_model = None
        best_params = None
        for h, dr in product(param_grid["hidden_size"], param_grid["dropout"]):
            try:
                train_ds = self._make_tsdataset(train_df, max_encoder_length, max_prediction_length)
                val_ds = self._make_tsdataset(test_df, max_encoder_length, max_prediction_length)
                train_dl = train_ds.to_dataloader(train=True, batch_size=32, num_workers=0)
                val_dl = val_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

                model = DeepAR.from_dataset(
                    train_ds,
                    hidden_size=h,
                    dropout=dr,
                    learning_rate=1e-2
                )
                trainer = Trainer(max_epochs=5, logger=False, enable_checkpointing=False, enable_progress_bar=False)
                trainer.fit(model, train_dl)
                preds = model.predict(val_dl).numpy().flatten()
                true = test_df.tail(max_prediction_length)["y"].values
                mae = mean_absolute_error(true, preds)
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_params = {"hidden_size": h, "dropout": dr}
            except Exception as e:
                continue
        self.monthly_model = best_model
        self.monthly_test_data = test_df.tail(max_prediction_length)
        self.best_monthly_params = best_params

    def forecast_future_daily(self, periods=7):
        # Use last context window
        df = self.daily_df.copy()
        df["ds"] = (df["ds"] - df["ds"].min()).dt.days
        last_context = df.iloc[-30:].copy()
        future_index = [last_context["ds"].max() + i + 1 for i in range(periods)]
        for reg in self.regressors:
            last_context[reg] = last_context[reg].iloc[-1]
        # Build new DataFrame for prediction
        prediction_data = last_context.copy()
        for i in range(periods):
            new_row = {col: last_context.iloc[-1][col] for col in last_context.columns}
            new_row["ds"] = future_index[i]
            new_row["y"] = np.nan
            prediction_data = pd.concat([prediction_data, pd.DataFrame([new_row])])
        tsdataset = self._make_tsdataset(prediction_data, 30, periods)
        loader = tsdataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        preds = self.daily_model.predict(loader).numpy().flatten()
        # Build output df with dates
        date_base = self.daily_df["ds"].max()
        future_dates = pd.date_range(start=date_base + pd.Timedelta(days=1), periods=periods, freq='D')
        return pd.DataFrame({"ds": future_dates, "deepar_forecast": preds})

    def forecast_future_monthly(self, periods=3):
        df = self.monthly_df.copy()
        df["ds"] = (df["ds"] - df["ds"].min()).dt.days
        last_context = df.iloc[-6:].copy()
        future_index = [last_context["ds"].max() + 30 * (i + 1) for i in range(periods)]
        for reg in self.regressors:
            last_context[reg] = last_context[reg].iloc[-1]
        prediction_data = last_context.copy()
        for i in range(periods):
            new_row = {col: last_context.iloc[-1][col] for col in last_context.columns}
            new_row["ds"] = future_index[i]
            new_row["y"] = np.nan
            prediction_data = pd.concat([prediction_data, pd.DataFrame([new_row])])
        tsdataset = self._make_tsdataset(prediction_data, 6, periods)
        loader = tsdataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        preds = self.monthly_model.predict(loader).numpy().flatten()
        date_base = self.monthly_df["ds"].max()
        future_dates = pd.date_range(start=date_base + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        return pd.DataFrame({"ds": future_dates, "deepar_forecast": preds})
