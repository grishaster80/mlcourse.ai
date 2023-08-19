import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.precision", 2)
DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
df = pd.read_csv(DATA_URL + "adult.data.csv")

# 1. How many men and women (sex feature) are represented in this dataset?
print(df["sex"].value_counts())

# 2. What is the average age (age feature) of women?
print(df[df["sex"] == "Female"]["age"].mean())

# 3. What is the percentage of German citizens (native-country feature)?
print(float((df["native-country"] == "Germany").sum()) / df.shape[0])

# 4-5. What are the mean and standard deviation of age for those who earn more than 50K per year (salary feature) and those who earn less than 50K per year?
ages1 = df[df["salary"] == ">50K"]["age"]
ages2 = df[df["salary"] == "<=50K"]["age"]
print(ages1.mean())
print(ages1.std())
print(ages2.mean())
print(ages2.std())

# 6. Is it true that people who earn more than 50K have at least high school education?
print(df[df["salary"] == ">50K"]["education"].unique())

# 7. Display age statistics for each race (race feature) and each gender (sex feature). Use groupby() and describe().
# Find the maximum age of men of Amer-Indian-Eskimo race.

for (race, sex), sub_df in df.groupby(["race", "sex"]):
    print("Race: {0}, sex: {1}".format(race, sex))
    print(sub_df["age"].describe())

# 8. Among whom is the proportion of those who earn a lot
df[(df["sex"] == "Male")
     & (df["marital-status"].str.startswith("Married"))][
    "salary"
].value_counts(normalize=True)

# single men
df[
    (df["sex"] == "Male")
    & ~(df["marital-status"].str.startswith("Married"))
]["salary"].value_counts(normalize=True)


# 9. What is the maximum number of hours a person works per week

max_load = df["hours-per-week"].max()
print("Max time - {0} hours./week.".format(max_load))

num_workaholics = df[df["hours-per-week"] == max_load].shape[0]
print("Total number of such hard workers {0}".format(num_workaholics))

rich_share = (
    float(
        df[(df["hours-per-week"] == max_load) & (df["salary"] == ">50K")].shape[0]
    )
    / num_workaholics
)
print("Percentage of rich among them {0}%".format(int(100 * rich_share)))

# 10. Count the average time of work (hours-per-week) those who earning a little and a lot

for (country, salary), sub_df in df.groupby(["native-country", "salary"]):
    print(country, salary, round(sub_df["hours-per-week"].mean(), 2))
