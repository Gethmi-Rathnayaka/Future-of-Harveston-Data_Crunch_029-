import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

#Load data sets
train_df = pd.read_csv("/kaggle/input/data-crunch-round-1/train.csv")
test_df = pd.read_csv("/kaggle/input/data-crunch-round-1/test.csv")
sample_sub = pd.read_csv("/kaggle/input/data-crunch-round-1/sample_submission.csv")

#Necessary columns
numeric_columns = ['latitude', 'longitude', 'Avg_Temperature', 'Avg_Feels_Like_Temperature', 'Temperature_Range',
                   'Feels_Like_Temperature_Range', 'Radiation', 'Rain_Amount', 'Rain_Duration', 'Wind_Speed',
                   'Wind_Direction', 'Evapotranspiration']
categorical_columns = ['kingdom']

def ensure_columns(df, columns):
    missing_cols = [col for col in columns if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan

ensure_columns(train_df, numeric_columns + categorical_columns)
ensure_columns(test_df, numeric_columns + categorical_columns)

#Missing values
train_df[numeric_columns] = train_df[numeric_columns].apply(lambda x: x.fillna(x.median()))
test_df[numeric_columns] = test_df[numeric_columns].apply(lambda x: x.fillna(train_df[x.name].median()))

for col in categorical_columns:
    mode_value = train_df[col].mode()[0]
    train_df[col].fillna(mode_value, inplace=True)
    test_df[col].fillna(mode_value, inplace=True)

#Function to handle 3-digit year format
def parse_3_digit_year(row):
    year = int(row['Year'])
    month = int(row['Month'])
    day = int(row['Day'])

    if year < 1000:  
        year = 1000 + year  

    if year < 1900 or year > 2100: 
        year = 2000  
    
    return f"{year:04d}-{month:02d}-{day:02d}"

for df in [train_df, test_df]:
    if all(col in df.columns for col in ['Year', 'Month', 'Day']):
        df['date'] = df.apply(parse_3_digit_year, axis=1)
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            print(f"Error in date conversion: {e}")
        df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df.drop(columns=['date'], inplace=True)

#Target and feature columns
target_columns = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']
feature_columns = [col for col in train_df.columns if col not in ['ID'] + target_columns]

#Prepare data
existing_target_columns = [col for col in target_columns if col in train_df.columns]
X_train = train_df[feature_columns]
y_train = train_df[existing_target_columns]
X_test = test_df.drop(columns=['ID'], errors='ignore')

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

#Imputation
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

X_train_small, X_val, y_train_small, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Model training
rf = RandomForestRegressor(n_estimators=100, random_state=42)
multi_target_model = MultiOutputRegressor(rf)
multi_target_model.fit(X_train_small, y_train_small)

y_val_pred = multi_target_model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Validation Mean Absolute Error: {mae}")

multi_target_model.fit(X_train, y_train)

y_test_pred = multi_target_model.predict(X_test)
y_test_pred_df = pd.DataFrame(y_test_pred, columns=existing_target_columns)

submission = test_df[['ID']].copy() if 'ID' in test_df.columns else pd.DataFrame()
for col in existing_target_columns:
    submission[col] = y_test_pred_df[col]
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission Preview:\n", submission.head())