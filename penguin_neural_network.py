# Here is an example of code to create a simple neural network using the Keras library in Python
# to classify the penguin species

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

# Load the data
penguin_df = pd.read_csv(r"data/penguin_raw_data.csv")
before_nadropped = penguin_df.shape[0]
penguin_df = penguin_df.dropna()
difference = before_nadropped - penguin_df.shape[0]
print(f"{difference} rows have been dropped")

penguin_df["Species"] = penguin_df["Species"].str.replace(" ", "")
penguin_df["Species"] = penguin_df["Species"].str.split("(").str[0]
penguin_df["Species"] = penguin_df["Species"].astype("category")
data_types = penguin_df.dtypes
print(data_types)

pen_df_numeric = pd.DataFrame()
for column in penguin_df:
    pen_df_numeric.append(column)
    if penguin_df[column].dtype == "object" or "category":
        # Fit the LabelEncoder to the column and transform it
        encoder = LabelEncoder()
        pen_df_numeric[column] = encoder.fit_transform(penguin_df[column])
print(penguin_df.dtypes)  # these should be the same as before
print(pen_df_numeric.dtypes)  # these should all be numeric
print(pen_df_numeric)


# Split the data into predictor and outcome variables
predictors = penguin_df.iloc[:, 3:]
outcome = penguin_df["Species"]



