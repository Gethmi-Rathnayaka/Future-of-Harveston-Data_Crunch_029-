import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

#Load data sets
train = pd.read_csv("/kaggle/input/data-crunch-round-1/train.csv")
test = pd.read_csv("/kaggle/input/data-crunch-round-1/test.csv")
sample_sub = pd.read_csv("/kaggle/input/data-crunch-round-1/sample_submission.csv")


def preprocess(df):
    df.loc[:, "Year"] = df["Year"].apply(lambda x: x + 2000 if x < 1000 else x)
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors='coerce')
    df = df.sort_values("Date")
    df.drop(columns=["Year", "Month", "Day"], inplace=True)
    
    
    if "kingdom" in df.columns:
        encoder = LabelEncoder()
        df["kingdom"] = encoder.fit_transform(df["kingdom"])
    
    return df

train = preprocess(train)
test = preprocess(test)

#Feature Engineering
if "Avg_Temperature" in train.columns:
    train["Temp_Lag1"] = train["Avg_Temperature"].shift(1)
if "Radiation" in train.columns:
    train["Radiation_Lag1"] = train["Radiation"].shift(1)
train.bfill(inplace=True)

#Target columns
targets = ["Avg_Temperature", "Radiation", "Rain_Amount", "Wind_Speed", "Wind_Direction"]
features = [col for col in train.columns if col not in ["ID", "Date"] + targets]

available_features = [col for col in features if col in test.columns]
X_train, X_val, y_train, y_val = train_test_split(train[available_features], train[targets], test_size=0.2, random_state=42)

models = {}
predictions = {}

for target in targets:
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42, enable_categorical=True)
    model.fit(X_train, y_train[target])
    y_pred = model.predict(X_val)
    print(f"sMAPE for {target}: {smape(y_val[target], y_pred):.2f}%")
    models[target] = model
    if set(available_features).issubset(test.columns):
        predictions[target] = model.predict(test[available_features])
    else:
        predictions[target] = np.zeros(len(test))

submission = test[["ID"].copy()]
for target in targets:
    submission[target] = predictions[target]

submission.to_csv("submission.csv", index=False)
print("Submission file saved!")