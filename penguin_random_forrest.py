# A script to identify species of penguins in the Palmer Archipelago using unsupervised and supervised ML methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ~ Clean the dataset
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


# ~ Exploratory Data Analysis

# create a basic plot to show the distribution of the outcomes
countplot_simple = sns.countplot(x="Species", data=penguin_df)
plt.show()

# Plot species count with viridis color palette
colors = sns.color_palette("viridis", len(penguin_df["Species"].value_counts()))
countplot_ordered = sns.countplot(
    x="Species",
    data=penguin_df,
    order=penguin_df["Species"].value_counts().index,
    palette=colors,
)
plt.xlabel("Species")
plt.ylabel("Count")
plt.title("Penguin Species Count")
plt.show()

# Create a pairplot to visualise relationships between different predictors in the dataset

penguin_pairplot = sns.pairplot(
    data=penguin_df.iloc[:, 2:],
    # select all rows and columns from index position 2 to the end
    hue="Species",
)
plt.show()
# Most of these correlations appear to segregate the groups, so we will continue to predictive models


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# The dataset needs preprocessing, the RandomForestClassifier algorithm requires all values to be numeric
from sklearn.preprocessing import LabelEncoder

for column in penguin_df:
    if penguin_df[column].dtype == "object" or "category":
        # Fit the LabelEncoder to the column and transform it
        encoder = LabelEncoder()
        penguin_df[column] = encoder.fit_transform(penguin_df[column])
print(penguin_df.dtypes)

# Select the variables to be the predictors or the outcome
predictors = penguin_df.iloc[:, 3:]
outcome = penguin_df["Species"]
print("Predictors have been selected")

# Create the train and test split
predictor_train, predictor_test, outcome_train, outcome_test = train_test_split(
    predictors, outcome, test_size=0.2, random_state=1
)

# Create the random forest classifier and train it
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(predictor_train, outcome_train)

# Make predictions on the test data and calculate accuracy of the model
outcome_predict = rf.predict(predictor_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(outcome_test, outcome_predict)
print(f"Accuracy: {accuracy}")

# Lets do a 5-fold cross validation to check the overall accurary of the model
cv_scores_total = cross_val_score(rf, predictors, outcome, cv=5)
print(f"Cross-Validation Scores: {cv_scores_total}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores_total)}\n")


# Amazing, the model can predict with 98.46% accuracy which species the penguin is from
# Since there are many variables, and some are probably not very predictive, it is useful to find out which
# Variables are most useful by checking feature importances

# Get feature importances
importances = rf.feature_importances_

# Get the index of the features sorted by importances
sorted_index = np.argsort(importances)[::-1]

# Print the feature importances
print("Feature importances:")
for index in sorted_index:
    print(f"{predictors.columns[index]}: {importances[index]}")
print(f"\n")

# Lets remove the bottom 3 least important predictors and re-run the regression
num_to_remove = 3
predictors_pruned = predictors.drop(predictors.columns[sorted_index[-num_to_remove:]], axis=1)

# Split the pruned dataset into train and test sets
predictor_pruned_train, predictor_pruned_test, outcome_pruned_train, outcome_pruned_test = train_test_split(
    predictors_pruned, outcome, test_size=0.2, random_state=2
)

# Create the random forrest regressor with the same parameters as before
rf_pruned = RandomForestClassifier(n_estimators=100, random_state=2)
rf_pruned.fit(predictor_pruned_train, outcome_pruned_train)

cv_scores_total_pruned = cross_val_score(rf_pruned, predictors_pruned, outcome, cv=5)
print(f"Cross-Validation Scores for Pruned Model: {cv_scores_total_pruned}")
print(f"Mean Cross-Validation Score for Pruned Model: {np.mean(cv_scores_total_pruned)}")
percent_diff = (1 - np.mean(cv_scores_total_pruned)/np.mean(cv_scores_total)) * 100
print(f"The pruned model is {percent_diff:.2f}% less accurate \n")

# Let's check which features have been removed
importances_pruned = rf_pruned.feature_importances_
sorted_index = np.argsort(importances_pruned)[::-1]
print("Feature importances for pruned model:")
for index in sorted_index:
    print(f"{predictors_pruned.columns[index]}: {importances_pruned[index]}")
print(f"\n")




