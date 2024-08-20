import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

#load the BRCA dataset with subtypes
brca_data = pd.read_csv("archive/brca_data_w_subtypes.csv")

#separate Numeric and Categorical Columns
numeric_cols = brca_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = brca_data.select_dtypes(include=['object']).columns

#remove the label column from categorical columns list
categorical_cols = categorical_cols.drop('histological.type')

#preprocessing Pipelines for Numeric and Categorical Data
#numeric pipeline, impute missing values with median and normalize
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#categorical pipeline, impute missing values with the most frequent value and one-hot encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#combine both pipelines
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

#apply the Preprocessor to the Dataset
X_preprocessed = preprocessor.fit_transform(brca_data)

#get feature names after one-hot encoding
feature_names = numeric_cols.tolist() + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols).tolist()
X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

#separate Features and Labels
#labels(target variable)
y = brca_data['histological.type']

#feature selection
#apply Variance Threshold to remove features with low variance
selector = VarianceThreshold(threshold=0.01)  
X_selected = pd.DataFrame(selector.fit_transform(X_preprocessed), columns=X_preprocessed.columns[selector.get_support()])

#split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

#save preprocessed data for future use (optional)
X_train.to_csv("X_train_preprocessed.csv", index=False)
X_test.to_csv("X_test_preprocessed.csv", index=False)
y_train.to_csv("y_train_preprocessed.csv", index=False)
y_test.to_csv("y_test_preprocessed.csv", index=False)

print("Preprocessing complete. Training and testing sets are ready.")
