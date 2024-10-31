# knn_classification.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and explore the dataset
def load_data():
    # Sample dataset: replace with actual dataset or load from a file
    data = {
        'Feature1': [5.1, 4.9, 4.7, 6.3, 5.8],
        'Feature2': [3.5, 3.0, 3.2, 3.3, 3.0],
        'Feature3': [1.4, 1.4, 1.3, 6.0, 5.1],
        'Feature4': [0.2, 0.2, 0.2, 2.5, 1.8],
        'Label': [0, 0, 0, 1, 1]  # Binary labels for simplicity
    }
    df = pd.DataFrame(data)
    return df

# Split data into training and test sets
def split_data(df):
    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Scale features for KNN
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train and evaluate KNN model
def train_and_evaluate_knn(X_train, X_test, y_train, y_test, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Determine optimal k using accuracy score
# Determine optimal k using accuracy score, ensuring k is <= number of samples in the training set
def find_optimal_k(X_train, X_test, y_train, y_test, max_k=10):
    accuracy = []
    # Adjust max_k to be the minimum of max_k or the number of samples in the training set
    max_k = min(max_k, len(X_train))
    
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        accuracy.append(score)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), accuracy, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Neighbors')
    plt.show()


# Main function to run all steps
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Determine optimal k using accuracy plot
    find_optimal_k(X_train_scaled, X_test_scaled, y_train, y_test)

    # Train and evaluate KNN with chosen k (set k=3 as an example)
    k = 3
    y_pred = train_and_evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, k=k)

    # Visualize performance with a confusion matrix
    plot_confusion_matrix(y_test, y_pred)
