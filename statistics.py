import pandas as pd

path = "data\\raw data\\GRF_F_V_RAW_right.csv"
df = pd.read_csv(path)

# Grouping by 'SUBJECT_ID' and counting trials
trial_counts = df.groupby("SUBJECT_ID").size()

# Calculating average, minimum, and maximum number of trials per participant
average_trials = trial_counts.mean()
min_trials = trial_counts.min()
max_trials = trial_counts.max()

print("Average number of trials per participant:", average_trials)
print("Minimum number of trials per participant:", min_trials)
print("Maximum number of trials per participant:", max_trials)


# Counting the number of trials for males and females
male_count = (df["SEX"] == 1).sum()
female_count = (df["SEX"] == 0).sum()

print("Number of trials performed by males:", male_count)
print("Number of trials performed by females:", female_count)


# Keep only unique rows per SUBJECT_ID
unique_subjects = df.drop_duplicates(subset="SUBJECT_ID")

# Counting the number of males and females
male_count = (unique_subjects["SEX"] == 1).sum()
female_count = (unique_subjects["SEX"] == 0).sum()

print("Number of males in the dataset:", male_count)
print("Number of females in the dataset:", female_count)
