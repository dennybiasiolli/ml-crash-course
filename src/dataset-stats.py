import pandas as pd

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# load the dataset
training_df = pd.read_csv(filepath_or_buffer="data/california_housing_train.csv")

print(training_df.describe())
# for column in training_df.columns:
#     print(f"""Column: {column}
#     * mean: {training_df[column].mean():.4f}
#     * standard deviation: {training_df[column].std():.4f}
#     * delta between min and 25%: {training_df[column].describe()['25%'] - training_df[column].describe()['min']:.4f}
#     * delta between 75% and max: {training_df[column].describe()['max'] - training_df[column].describe()['75%']:.4f}""")
print("""The following columns might contain outliers:

  * total_rooms
  * total_bedrooms
  * population
  * households
  * possibly, median_income

In all of those columns:

  * the standard deviation is almost as high as the mean
  * the delta between 75% and max is much higher than the
      delta between min and 25%.""")
