import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Define the correct dataset path
file_path = r"D:\Health insurance prediction\insurance.csv"  # Ensure this file exists
df = pd.read_csv(file_path)

# Handle missing values (if any)
df = df.dropna()

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['sex', 'smoker', 'region']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for later decoding if needed

# Scaling numerical features
scaler = StandardScaler()
numerical_columns = ['age', 'bmi', 'children']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Define the new output directory
output_dir = r"D:\Health insurance prediction"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Save cleaned dataset
cleaned_file_path = os.path.join(output_dir, "cleaned_insurance.csv")
df.to_csv(cleaned_file_path, index=False)

print(f"Preprocessed data saved to: {cleaned_file_path}")
