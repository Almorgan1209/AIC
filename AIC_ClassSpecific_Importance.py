# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:56:58 2023

@author: almor
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from scipy import stats

#Import filtered datasets with latent class assignments
academic_df = pd.read_csv(r"LatentClasses_academic.csv")
health_df = pd.read_csv(r"LatentClasses_health.csv")

##############
#Look at information use in latent classes.
# Function to calculate feature importances within latent classes
def calculate_feature_importances_within_classes(df, features, target, n_folds=5):
    importances_array = np.zeros((n_folds, len(features)))
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=123)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=123)
    
    for fold, (train_index, test_index) in enumerate(cv.split(df[features], df[target])):
        train_data = df.iloc[train_index]
        gb.fit(train_data[features], train_data[target])
        importances = gb.feature_importances_
        importances_array[fold] = importances

    mean_importances = np.mean(importances_array, axis=0)
    std_importances = np.std(importances_array, axis=0)
    alpha = 0.05
    n_samples = len(importances_array)
    t_critical = abs(stats.t.ppf(alpha / 2, n_samples - 1))
    conf_int = t_critical * std_importances / np.sqrt(n_samples)
    lower_bounds = mean_importances - conf_int
    upper_bounds = mean_importances + conf_int

    feature_importance_results = []
    for i, feature in enumerate(features):
        feature_result = {
            "Feature": feature,
            "Mean Importance": mean_importances[i],
            "Lower Bound": lower_bounds[i],
            "Upper Bound": upper_bounds[i]
        }
        feature_importance_results.append(feature_result)

    return feature_importance_results

# Function to count class frequencies and total samples
def count_class_frequencies(df):
    class_counts = df.groupby('LatentClass')['ID'].unique().apply(lambda x: len(x))
    total_samples = len(class_counts)
    return class_counts, total_samples

# Function to analyze descriptive differences between classes
def class_feature_differences(df):
    class_vars = ['LatentClass']
    descriptive_results = {}
    
    # Calculate frequency distribution for latent classes
    for var in class_vars:
        freq_table = df[var].value_counts().to_dict()
        descriptive_results[var] = freq_table
    
    # Calculate mode for feature columns for each latent class
    features_to_analyze = ['Author', 'Medium', 'Structure']
    for feature in features_to_analyze:
        mode_by_class = df.groupby('LatentClass')[feature].apply(lambda x: x.mode().values[0]).to_dict()
        descriptive_results[feature] = mode_by_class
    
    # Get the name of the DataFrame
    df_name = [name for name, df_ in globals().items() if df_ is df][0] 

    # Print modes for each feature within each latent class
    for latent_class, modes in descriptive_results['Author'].items():
        print(f'{df_name} Latent Class {latent_class}')
        print(f'Author: {modes}')
        print(f'Medium: {descriptive_results["Medium"].get(latent_class, "N/A")}')
        print(f'Structure: {descriptive_results["Structure"].get(latent_class, "N/A")}')
        print()

    
    return descriptive_results

# Function to rename attribute levels
def rename_attribute_levels(df):
    df['Author'] = df['Author'].replace({1: 'Expert', 2: 'Layman'})
    df['Medium'] = df['Medium'].replace({1: 'News', 2: 'Academic', 3: 'Direct', 4: 'Online', 5: 'Social Media'})
    df['Structure'] = df['Structure'].replace({1: 'Facts and Numbers', 2: 'Personal Story about their Experience'})
    return df

# Example usage for health and academic latent classes dataframes
health_latent_class_features = ['Author', 'Medium', 'Structure']
academic_latent_class_features = ['Author', 'Medium', 'Structure']

health_feature_importances = calculate_feature_importances_within_classes(health_df, health_latent_class_features, 'Selected')
academic_feature_importances = calculate_feature_importances_within_classes(academic_df, academic_latent_class_features, 'Selected')

health_class_counts, health_total_samples = count_class_frequencies(health_df)
academic_class_counts, academic_total_samples = count_class_frequencies(academic_df)

health_df = rename_attribute_levels(health_df)
academic_df = rename_attribute_levels(academic_df)

health_descriptive_differences = class_feature_differences(health_df)
academic_descriptive_differences = class_feature_differences(academic_df)



# Print results
print("Health Feature Importances within Latent Classes:")
print(health_feature_importances)
print("Academic Feature Importances within Latent Classes:")
print(academic_feature_importances)

print("Health Class Frequencies and Total Samples:")
print(health_class_counts)
print("Total number of samples for health: ", health_total_samples)

print("Academic Class Frequencies and Total Samples:")
print(academic_class_counts)
print("Total number of samples for academic: ", academic_total_samples)

#Compare whether Latent Class Assignment is related between topics.
def chi_square_membership(health_df, academic_df):
    # Extract the ID columns
    health_ids = health_df["ID"]
    academic_ids = academic_df["ID"]

    # Create a new dataframe with ID and LatentClass columns
    health_latent_class = pd.DataFrame({"ID": health_ids, "LatentClass": health_df["LatentClass"]})
    academic_latent_class = pd.DataFrame({"ID": academic_ids, "LatentClass": academic_df["LatentClass"]})

    # Merge the dataframes on the ID column
    merged_df = pd.merge(health_latent_class, academic_latent_class, on="ID")

    # Create the contingency table and perform the chi-square test
    contingency_table = pd.crosstab(merged_df["LatentClass_x"], merged_df["LatentClass_y"])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Print the results
    print("Chi-square statistic:", chi2)
    print("p-value:", p_value)
    print("Degrees of freedom:", dof)
    print("Expected frequencies:", expected)

# Example usage
chi_square_membership(health_df, academic_df)
