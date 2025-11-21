import pandas as pd
from matplotlib import pyplot as plt
import io

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# load the dataset, (50 students, evaluated on 28 consecutive days)
# rows 0-49 show how each of the students performed on the first day,
# rows 50-99 show how each of the students performed on the second day,
# and so on
training_df = pd.read_csv(filepath_or_buffer="data/calories_test_score.csv")

print(training_df.describe())
# for column in training_df.columns:
#     print(f"""Column: {column}
#     * mean: {training_df[column].mean():.4f}
#     * standard deviation: {training_df[column].std():.4f}
#     * delta between min and 25%: {training_df[column].describe()['25%'] - training_df[column].describe()['min']:.4f}
#     * delta between 75% and max: {training_df[column].describe()['max'] - training_df[column].describe()['75%']:.4f}""")
print("""The basic statistics do not suggest a lot of outliers.
The standard deviations are substantially less than the
means. Furthermore, the quartile boundaries are approximately
evenly spaced.""")

def plot_the_dataset(feature, label, number_of_points_to_plot):
  """Plot N random points of the dataset."""
  # Label the axes.
  plt.xlabel(feature)
  plt.ylabel(label)
  # Create a scatter plot from n random points of the dataset.
  random_examples = training_df.sample(n=number_of_points_to_plot)
  plt.scatter(random_examples[feature], random_examples[label])
  # Render the scatter plot.
  plt.show()

def plot_a_contiguous_portion_of_dataset(feature, label, start, end):
  """Plot the data points from start to end."""
  # Label the axes.
  plt.xlabel(feature + "Day")
  plt.ylabel(label)
  # Create a scatter plot.
  plt.scatter(training_df[feature][start:end], training_df[label][start:end])
  # Render the scatter plot.
  plt.show()

plot_the_dataset("calories", "test_score", number_of_points_to_plot=50)
plot_the_dataset("calories", "test_score", number_of_points_to_plot=len(training_df.index))
print("""Visualizing 50 data points doesn't imply any outliers.
However, as you increase the number of random data points to plot, a
clump of outliers appears. Notice the points with high test scores but less
than 200 calories.""")

# Get statistics by week (50 students * 7 days = 350 rows)
for i in range(4):
    start_index = i * 350
    end_index = start_index + 350
    print(f"Statistics for week {i+1}:")
    print(training_df[start_index:end_index].describe())
print("""The basic statistics for each week are pretty similar, so weekly
differences aren't a likely explanation for the outliers.""")

# Visualize by day of week
for i in range(7):
    start_index = i * 50
    end_index = start_index + 50
    plot_a_contiguous_portion_of_dataset("calories", "test_score", start_index, end_index)
print("""Wait a second--the calories value for Day 4 spans 0 to 200, while the
calories value for all the other Days spans 0 to 400. Something is wrong
with Day 4, at least on the first week.""")

running_total_of_thursday_calories = 0
running_total_of_non_thursday_calories = 0
count = 0
for week in range(0,4):
  for day in range(0,7):
    for subject in range(0,50):
      position = (week * 350) + (day * 50) + subject
      if (day == 4):  # Thursday
        running_total_of_thursday_calories += training_df['calories'][position]
      else:  # Any day except Thursday
        count += 1
        running_total_of_non_thursday_calories += training_df['calories'][position]

mean_of_thursday_calories = running_total_of_thursday_calories / 200
mean_of_non_thursday_calories = running_total_of_non_thursday_calories / 1200

print("The mean of Thursday calories is %.0f" % (mean_of_thursday_calories))
print("The mean of calories on days other than Thursday is %.0f" % (mean_of_non_thursday_calories))
