# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:05:52 2023

@author: almor
"""
import pandas as pd
from scipy import stats

academic_membership = pd.read_csv(r"\LatentClasses_academic.csv")
health_membership = pd.read_csv(r"\LatentClasses_health.csv")

# Merge the data frames based on the "ID" column to get common participants
common_participants = pd.merge(
    health_membership, academic_membership, on="ID")["ID"]

# Filter the original data frames to include only the common participants
common_participants = set(health_membership["ID"]).intersection(academic_membership["ID"])

filtered_health_membership = health_membership[health_membership["ID"].isin(common_participants)]
filtered_academic_membership = academic_membership[academic_membership["ID"].isin(common_participants)]

# Group the data by "ID" and select the first occurrence of each participant
grouped_health_membership = filtered_health_membership.groupby("ID").first()
grouped_academic_membership = filtered_academic_membership.groupby("ID").first()

# Create the contingency table and perform the chi-square test
contingency_table = pd.crosstab(grouped_health_membership["LatentClass"], grouped_academic_membership["LatentClass"])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)