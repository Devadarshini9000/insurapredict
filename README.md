# 🏥 Insurapredict - Health Insurance Prediction Web App

A **Flask-based machine learning web application** that predicts **health insurance eligibility** and **cost category** based on user inputs like age, BMI, number of children, smoker status, and either region or income. The app also provides **data visualizations** for deeper insights into health insurance datasets.

---

## 🚀 Features

- 🤖 **ML-powered eligibility prediction** (Random Forest + PCA + Scaler)
- 📊 Predicts **insurance cost levels**: Low, Medium, High, Very High
- 🧮 Accepts **region** or **income** for flexible predictions
- 📈 Displays visual insights:
  - Correlation matrix
  - BMI vs Insurance Charges scatter plot
  - Age group vs Charges bar chart
  - Smokers vs Non-Smokers cost comparison
- 🖥️ **User-friendly Flask interface**

---

## 📁 Project Structure

├── app.py # Flask application
├── preprocess.py # Data preprocessing script
├── train.py # Model training script
├── eda.py # Exploratory Data Analysis
├── best_model.pkl # Trained ML model
├── scaler.pkl # Scaler object for numerical features
├── pca.pkl # PCA object for dimensionality reduction
├── static/ # Generated visualization images
└── templates/index.html # Frontend HTML template

---

## 🛠 Tech Stack

- **Backend**: Python, Flask
- **ML & Data Processing**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Local Flask server

---

## 🧪 How It Works

1. **Data Preprocessing (`preprocess.py`)**
   - Cleans missing values
   - Encodes categorical features (`sex`, `smoker`, `region`)
   - Scales numerical features (`age`, `bmi`, `children`)
   - Saves processed dataset as `cleaned_insurance.csv`

2. **Exploratory Data Analysis (`eda.py`)**
   - Generates feature correlations
   - Visualizes relationships between BMI, age, smoking, and charges

3. **Model Training (`train.py`)**
   - Applies PCA for dimensionality reduction
   - Trains multiple models (Random Forest, SVM, Logistic Regression, etc.)
   - Selects the best model based on F1 score
   - Saves `best_model.pkl`, `scaler.pkl`, `pca.pkl`

4. **Prediction & UI (`app.py`)**
   - Accepts user input via web form
   - Predicts eligibility and cost category
   - Displays risk factors or rejection reasons
   - Shows visual insights

---

## 💻 How to Run Locally

### 1️⃣ Clone the Repository
git clone https://github.com/yourusername/health-insurance-predictor.git
cd health-insurance-predictor
2️⃣ Install Dependencies
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
3️⃣ Preprocess Data
python preprocess.py
4️⃣ Train Model
python train.py
5️⃣ Run the Web App
python app.py

🎯 Use Cases
-🏥 Insurance companies estimating eligibility and premium ranges
-📊 Health data analysis for trends and risk factors
-🎓 ML portfolio project for predictive analytics

## 👩‍💻 Author
**Devadarshini P**  
[🔗 LinkedIn](https://www.linkedin.com/in/devadarshini-p-707b15202/)  
[💻 GitHub](https://github.com/Devadarshini9000)

“Predict smart, insure right.” – Health Insurance Predictor
