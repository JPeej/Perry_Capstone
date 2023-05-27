import numpy as np
import pandas as pd
from scipy import stats
from sklearn import model_selection
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor


class Model:
    # DATA HANDLING
    # Get dataset
    dataset = "Resources/bodyfat.csv"
    df = pd.read_csv(dataset)

    # Drop unnecessary columns
    df.drop(labels="Density", axis=1, inplace=True)

    # Delete any rows with null values, no values are found but good practice if CSV is updated
    df.dropna()

    # Delete any duplicate entries, no values are found but good practice if CSV is updated
    df.drop_duplicates()

    # Drop entries with an outlier value
    z_score = np.abs(stats.zscore(df))
    df = df[(z_score < 3).all(axis=1)]

    # Remove measurements that have weak correlation with bodyfat
    df.drop(columns=['Height', 'Ankle', 'Wrist', 'Age'], axis=1, inplace=True)

    # DEVELOP  MODEL
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    # Get dependent(y, bodyfat) and independent(x, measurements) variables from split data
    bodyfatIndex = df.columns.get_loc("BodyFat")
    x_measurements = df.values[:, bodyfatIndex + 1:]
    y_bodyfat = df.values[:, bodyfatIndex]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_measurements, y_bodyfat, test_size=0.2)

    # Apply Yeo-Johnson Transformer
    pt = PowerTransformer()
    x_train = pt.fit_transform(x_train)
    x_test = pt.transform(x_test)

    # Train the model
    regressor.fit(x_train, y_train)
