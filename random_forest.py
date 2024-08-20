import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

#load the preprocessed training and testing data
X_train = pd.read_csv("X_train_preprocessed.csv")
X_test = pd.read_csv("X_test_preprocessed.csv")
y_train = pd.read_csv("y_train_preprocessed.csv").values.ravel()  #convert to 1D array
y_test = pd.read_csv("y_test_preprocessed.csv").values.ravel()  #convert to 1D array

#train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#make Predictions
y_pred = clf.predict(X_test)

#evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#create a directory for saving visualizations
visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

#confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(os.path.join(visualization_dir, "random_forest_confusion_matrix.png"))
plt.close() 

#feature Importance
importances = clf.feature_importances_
indices = np.argsort(importances)[-10:]  #top 10 important features

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.savefig(os.path.join(visualization_dir, "random_forest_feature_importance.png"))
plt.close() 

#PCA Visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette='Set1')
plt.title("PCA of Test Set")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig(os.path.join(visualization_dir, "random_forest_pca_visualization.png"))
plt.close()  

print("Model development and evaluation complete. Visualizations saved in the 'visualizations' folder.")
