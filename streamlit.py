import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import joblib

# Title of the Streamlit app
st.title("Classification Model for cityu10c_train_dataset")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Load the dataset
data = pd.read_csv('cityu10c_train_dataset.csv')

# Drop the ApplicationDate column
if 'ApplicationDate' in data.columns:
    data = data.drop(columns=['ApplicationDate'])

# Display the dataset in Streamlit
st.write("## Dataset Preview")
st.write(data.head())

# Handle missing values (if any)
data = data.dropna()

# Separate features and target variable
# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model and preprocessor to .pkl files
joblib.dump(model, 'trained_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the results in Streamlit
st.write("## Model Evaluation")
st.write("### Model Accuracy")
st.write(accuracy)
st.write("### Classification Report")
st.text(report)

# Add a section for user input to make predictions
st.write("## Make Predictions")
user_input = {}
for col in X.columns:
    if col in numerical_cols:
        user_input[col] = st.sidebar.number_input(f"Input {col}", value=float(X[col].mean()))
    else:
        user_input[col] = st.sidebar.selectbox(f"Select {col}", options=X[col].unique())

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Load the saved preprocessor and preprocess the user input
preprocessor = joblib.load('preprocessor.pkl')
input_processed = preprocessor.transform(input_df)

# Load the saved model and make predictions
model = joblib.load('trained_model.pkl')
prediction = model.predict(input_processed)

# Display the prediction result
st.write("### Prediction Result")
st.write(prediction[0])