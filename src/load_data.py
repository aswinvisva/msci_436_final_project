import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self):
        self._categorical_cols=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

    def transform_row(self, row):
        """Transform a single row from the saved scalers"""
        columns = [
            "Age",
            "Sex",
            "ChestPainType",
            "RestingBP",
            "Cholesterol",
            "FastingBS",
            "RestingECG",
            "MaxHR",
            "ExerciseAngina",
            "Oldpeak",
            "ST_Slope"
        ]

        with open('categorial_scalers.pkl', 'rb') as handle:
            lbl = pickle.load(handle)

        with open('regression_scaler.pkl', 'rb') as handle:
            scaler = pickle.load(handle)

        for i in self._categorical_cols:
            row[i]=lbl[i].transform(np.array([row[i]]).reshape(1, -1))[0]

        row = np.array([row[i] for i in columns]).reshape(1, -1)
        row = scaler.transform(row)

        return row

    def load(self, test_size=0.2, random_state=0):
        """Load and preprocess the data"""
        data = pd.read_csv('heart.csv')
        data.dropna(inplace=True)

        lbl={category: LabelEncoder() for category in self._categorical_cols}

        for i in self._categorical_cols:
            data[i]=lbl[i].fit_transform(data[i])

        with open('categorial_scalers.pkl', 'wb') as handle:
            pickle.dump(lbl, handle)

        target=data['HeartDisease']
        x_data=data.drop('HeartDisease',axis=1)

        X_train, X_test, y_train, y_test = train_test_split(x_data, target, test_size=test_size, random_state=random_state, shuffle=True)

        scaler = MinMaxScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)

        with open('regression_scaler.pkl', 'wb') as handle:
            pickle.dump(scaler, handle)

        return X_train, X_test, y_train, y_test
