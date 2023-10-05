# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# Load datasets
health_df = pd.read_csv(
    r"C:\Users\almor\OneDrive\Coding\health_conjoint_knowledge_removed.csv")
academic_df = pd.read_csv(
    r"C:\Users\almor\OneDrive\Coding\academic_conjoint_knowledge_removed.csv")

# Evaluate multinomial logit model
def evaluate_multinomial_logit(df, features=['Author', 'Medium', 'Structure'], target='Selected', task='Task'):
    onehot_encoder = OneHotEncoder(sparse_output=False)
    X = onehot_encoder.fit_transform(df[features])
    y = df[target]
    task = df[task]
    train_index = (task % 2 == 0)
    test_index = (task % 2 == 1)
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    mnl = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    mnl.fit(X_train, y_train)
    y_pred = mnl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, mnl.predict_proba(X_test))
    return X_train, X_test, y_train, y_test, accuracy, logloss

# Function to train and evaluate decision tree model
def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Function to train and evaluate random forest model
def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, max_depth=3)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return accuracy, precision, recall, f1, rf

# Function to calculate feature importances
def calculate_feature_importances(df, features, target):
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=123)
    importances = np.zeros(len(features))
    for _, fold in KFold(n_splits=5, shuffle=True, random_state=123).split(df[features], df[target]):
        train_data = df.iloc[fold]
        gb.fit(train_data[features], train_data[target])
        importances += gb.feature_importances_
    importances /= 5
    return importances

# Evaluate models
health_X_train, health_X_test, health_y_train, health_y_test, health_accuracy, health_logloss = evaluate_multinomial_logit(health_df)
academic_X_train, academic_X_test, academic_y_train, academic_y_test, academic_accuracy, academic_logloss = evaluate_multinomial_logit(academic_df)

# Train and evaluate decision tree models
h_tree_accuracy, h_tree_precision, h_tree_recall, h_tree_f1 = train_and_evaluate_decision_tree(health_X_train, health_y_train, health_X_test, health_y_test)
a_tree_accuracy, a_tree_precision, a_tree_recall, a_tree_f1 = train_and_evaluate_decision_tree(academic_X_train, academic_y_train, academic_X_test, academic_y_test)

# Train and evaluate random forest models
h_rf_accuracy, h_rf_precision, h_rf_recall, h_rf_f1, h_rf = train_and_evaluate_random_forest(health_X_train, health_y_train, health_X_test, health_y_test)
a_rf_accuracy, a_rf_precision, a_rf_recall, a_rf_f1, a_rf = train_and_evaluate_random_forest(academic_X_train, academic_y_train, academic_X_test, academic_y_test)

# Get feature importances
health_feature_importances = calculate_feature_importances(health_df, features=['Author', 'Medium', 'Structure'], target='Selected')
academic_feature_importances = calculate_feature_importances(academic_df, features=['Author', 'Medium', 'Structure'], target='Selected')

# Print results
print(f'Health Model Accuracy: {health_accuracy:.2f}')
print(f'Health Model Log Loss: {health_logloss:.2f}')

print(f'Academic Model Accuracy: {academic_accuracy:.2f}')
print(f'Academic Model Log Loss: {academic_logloss:.2f}')

print("Health Tree Metrics:")
print(f"Accuracy: {h_tree_accuracy}")
print(f"Precision: {h_tree_precision}")
print(f"Recall: {h_tree_recall}")
print(f"F1-Score: {h_tree_f1}")

print("\nAcademic Tree Metrics:")
print(f"Accuracy: {a_tree_accuracy}")
print(f"Precision: {a_tree_precision}")
print(f"Recall: {a_tree_recall}")
print(f"F1-Score: {a_tree_f1}")

print("Health Random Forest Metrics:")
print(f"Accuracy: {h_rf_accuracy:.2f}")
print(f"Precision: {h_rf_precision:.2f}")
print(f"Recall: {h_rf_recall:.2f}")
print(f"F1-Score: {h_rf_f1:.2f}")

print("Academic Random Forest Metrics:")
print(f"Accuracy: {a_rf_accuracy:.2f}")
print(f"Precision: {a_rf_precision:.2f}")
print(f"Recall: {a_rf_recall:.2f}")
print(f"F1-Score: {a_rf_f1:.2f}")

print("Health Feature Importances:")
print(health_feature_importances)
print("Academic Feature Importances:")
print(academic_feature_importances)

############################
# Function for Latent Class Analysis
def run_lca(name, X_train, X_test):
    
    # Lists to store fit statistics
    log_likelihoods = []
    bic_values = []
    silhouette_scores = []

    # Run Latent Class Analysis
    n_components = [2, 3, 4, 5, 6]
    for n in n_components:
        # Fit LCA analysis
        onehot_encoder = OneHotEncoder(sparse_output=False)
        X_train_encoded = onehot_encoder.fit_transform(X_train)
        X_test_encoded = onehot_encoder.transform(X_test)
        
        lca = GaussianMixture(n_components=n, random_state=123)
        lca.fit(X_train_encoded)
        
        # Evaluate the model
        log_likelihood = lca.score(X_test_encoded)
        bic = lca.bic(X_test_encoded)
        silhouette = silhouette_score(X_train_encoded, lca.predict(X_train_encoded))
                
        # Store fit statistics
        log_likelihoods.append(log_likelihood)
        bic_values.append(bic)
        silhouette_scores.append(silhouette)
        
        # Print fit statistics if needed
        print(f"{name} Model with {n} components")
        print(f"Log-likelihood: {log_likelihood:.3f}")
        print(f"BIC: {bic:.3f}")
        print(f"Silhouette Score: {silhouette:.2f}")
        print()
            
    # Plotting
    # Plot elbow curve based on log likelihoods
    plt.plot(n_components, log_likelihoods, 'o-')
    plt.xticks(np.arange(min(n_components), max(n_components)+1, 1))
    plt.xlabel('Number of Classes')
    plt.ylabel('Log-Likelihood')
    plt.title(f'Elbow Plot of Log-Likelihoods by Number of Classes for {name}')
    plt.show()
    
    # Plot based on silhouettes
    plt.plot(n_components, silhouette_scores, 'o-')
    plt.xlabel('Number of Classes')
    plt.ylabel('Silhouette Scores')
    plt.title(f'Elbow Plot of Silhouette Scores by Number of Classes for {name}')
    plt.ylim(bottom=0, top=0.50)
    plt.xticks(range(min(n_components), max(n_components)+1))
    plt.show()
    
    # Plot BIC values
    plt.plot(n_components, bic_values, 'o-', label='BIC')
    plt.xlabel('Number of Classes')
    plt.ylabel('BIC Value')
    plt.title(f'Plot of BIC Values by Number of Classes for {name}')
    plt.show()
        
# Example usage
run_lca('health_df', health_X_train, health_X_test)
run_lca('academic_df', academic_X_train, academic_X_test)

# Function to predict latent classes
def predict_latent_classes(X, n_classes):
    np.random.seed(123)
    gmm = GaussianMixture(n_components=n_classes, random_state=123)
    gmm.fit(X)
    participant_labels = []
    for participant_id in X.index.unique():
        participant_data = X.loc[X.index == participant_id]
        posterior_probs = gmm.predict_proba(participant_data)
        participant_labels.append(posterior_probs.argmax(axis=1) + 1)
    participant_labels = np.concatenate(participant_labels)
    return participant_labels

# Function to create a dataframe with latent class assignments
def create_latent_class_dataframe(df, participant_labels):
    df_new = pd.DataFrame({'ID': df['ID'], 'LatentClass': participant_labels})
    df_new.set_index('ID', inplace=True)
    participant_filtered_df = df_new.groupby('ID').first()
    return participant_filtered_df

# Function to merge and save latent classes
def merge_and_save_latent_classes(df, participant_filtered_df, file_name, output_path):
    filtered_df_attributes = df[['ID', 'Author', 'Medium', 'Structure', 'Selected']].join(
        participant_filtered_df['LatentClass'], on='ID')
    
    if "Health" in file_name:
        output_file = f"{output_path}/LatentClasses_health.csv"
    elif "Academic" in file_name:
        output_file = f"{output_path}/LatentClasses_academic.csv"
    
    filtered_df_attributes.to_csv(output_file, index=False)

# Predict latent classes
health_participant_labels = predict_latent_classes(health_df[['Author', 'Medium', 'Structure']], n_classes=2)
academic_participant_labels = predict_latent_classes(academic_df[['Author', 'Medium', 'Structure']], n_classes=2)

# Create dataframes with latent class assignments
health_participant_filtered_df = create_latent_class_dataframe(health_df, health_participant_labels)
academic_participant_filtered_df = create_latent_class_dataframe(academic_df, academic_participant_labels)

# Merge and save latent classes
merge_and_save_latent_classes(health_df, health_participant_filtered_df, "Health_Completed_df",r'C:\Users\almor\OneDrive\Coding\Portfolio\AIC')
merge_and_save_latent_classes(academic_df, academic_participant_filtered_df, "Academic_Completed_df", r'C:\Users\almor\OneDrive\Coding\Portfolio\AIC')


print(health_participant_filtered_df['LatentClass'].value_counts())
print(academic_participant_filtered_df['LatentClass'].value_counts())
