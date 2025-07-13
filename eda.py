import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"D:\Health insurance prediction\Dataset\insurance.csv")

# Encode categorical variables
categorical_columns = ['sex', 'smoker', 'region']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for reference if needed

# Summary statistics
print(df.describe())

# Pairplot to visualize relationships
sns.pairplot(df, hue="smoker")
plt.show()

# Correlation Heatmap (Now with Encoded Categories)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
