import os
import numpy as np
import pandas as pd
import joblib
import traceback
# Set Matplotlib backend to Agg (thread-safe) before other imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Define regions (as per dataset)
regions = ["Northeast", "Northwest", "Southeast", "Southwest"]

# Flag to track if models loaded successfully
models_loaded = True

# Load trained model and preprocessing objects with detailed error handling
try:
    model = joblib.load("best_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    traceback.print_exc()
    model = None
    models_loaded = False

try:
    scaler = joblib.load("scaler.pkl")
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    traceback.print_exc()
    scaler = None
    models_loaded = False

try:
    pca = joblib.load("pca.pkl")
    print("PCA loaded successfully")
except Exception as e:
    print(f"Error loading PCA: {str(e)}")
    traceback.print_exc()
    pca = None
    models_loaded = False

# Load dataset for visualization with better error handling
df = None
try:
    df = pd.read_csv(r"D:\Health insurance prediction\Dataset\cleaned_insurance.csv")
    print("CSV loaded from relative path")
except FileNotFoundError:
    try:
        # Fallback to a relative path if the above fails
        current_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(current_dir, "Dataset/cleaned_insurance.csv"))
        print("CSV loaded from absolute path")
    except FileNotFoundError:
        try:
            # Try one more path variation
            df = pd.read_csv(r"D:\Health insurance prediction\Dataset\cleaned_insurance.csv")
            print("CSV loaded from hard-coded absolute path")
        except Exception as e:
            print(f"All attempts to load CSV failed: {str(e)}")
            # Create minimal dummy dataset for visualizations
            df = pd.DataFrame({
                'age': range(20, 70, 5),
                'bmi': [25 + i/10 for i in range(10)],
                'children': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                'smoker': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                'charges': [5000 + i*2000 for i in range(10)]
            })
            print("Created dummy dataset for visualizations")

# Pre-generate visualizations at startup
corr_matrix_path = None
scatter_plot_path = None
age_chart_path = None
smoker_chart_path = None

# Function to create visualizations
def generate_visualizations():
    global corr_matrix_path, scatter_plot_path, age_chart_path, smoker_chart_path
    
    try:
        # Create a directory for static files if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Correlation Matrix
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix", fontsize=16, fontweight='bold')
        plt.tight_layout()
        corr_matrix_path = "static/corr_matrix.png"
        plt.savefig(corr_matrix_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Scatter Plot (BMI vs Charges)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df["bmi"], y=df["charges"], hue=df["smoker"], 
                       palette=["#3498db", "#e74c3c"], s=100, alpha=0.7)
        plt.xlabel("BMI", fontsize=14)
        plt.ylabel("Insurance Charges", fontsize=14)
        plt.title("BMI vs Insurance Charges", fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_plot_path = "static/scatter_plot.png"
        plt.savefig(scatter_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Age vs Charges Bar Chart
        plt.figure(figsize=(10, 6))
        age_groups = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 100])
        age_charges = df.groupby(age_groups)['charges'].mean().reset_index()
        sns.barplot(x=age_charges['age'].astype(str), y=age_charges['charges'], palette="viridis")
        plt.xlabel("Age Group", fontsize=14)
        plt.ylabel("Average Charges", fontsize=14)
        plt.title("Average Insurance Charges by Age Group", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        age_chart_path = "static/age_chart.png"
        plt.savefig(age_chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Smoker vs Non-Smoker Charges Comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='smoker', y='charges', data=df, palette=["#3498db", "#e74c3c"])
        plt.xlabel("Smoker Status", fontsize=14)
        plt.ylabel("Insurance Charges", fontsize=14)
        plt.title("Insurance Charges: Smokers vs Non-Smokers", fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        smoker_chart_path = "static/smoker_chart.png"
        plt.savefig(smoker_chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print("All visualizations generated successfully")
        
    except Exception as e:
        print(f"Error in generating visualizations: {str(e)}")
        traceback.print_exc()
        # Set default paths if visualization fails
        corr_matrix_path = "static/corr_matrix.png"
        scatter_plot_path = "static/scatter_plot.png"
        age_chart_path = "static/age_chart.png"
        smoker_chart_path = "static/smoker_chart.png"
    
    return corr_matrix_path, scatter_plot_path, age_chart_path, smoker_chart_path

# Generate visualizations at startup, not during request handling
try:
    corr_matrix_path, scatter_plot_path, age_chart_path, smoker_chart_path = generate_visualizations()
except Exception as e:
    print(f"Failed to generate visualizations: {str(e)}")
    corr_matrix_path = "static/corr_matrix.png"
    scatter_plot_path = "static/scatter_plot.png"
    age_chart_path = "static/age_chart.png"
    smoker_chart_path = "static/smoker_chart.png"

# Improved helper function for making predictions
def make_prediction(user_data):
    """Make a prediction with improved error handling and logic"""
    try:
        # Extract features from user_data for easier reference
        age = user_data[0][0]
        bmi = user_data[0][1]
        children = user_data[0][2]
        smoker = user_data[0][3]
        
        # If the ML model is available, use it
        if models_loaded and model is not None and scaler is not None and pca is not None:
            print("Using trained ML model for prediction")
            try:
                user_data_scaled = scaler.transform(user_data)
                user_data_pca = pca.transform(user_data_scaled)
                result = model.predict(user_data_pca)[0]
                print(f"ML model prediction: {result}")
                return result
            except Exception as e:
                print(f"Error using ML model: {str(e)}")
                # Fall back to rule-based approach if model fails
                print("Falling back to rule-based approach")
        else:
            print("Using rule-based approach (models not available)")
        
        # Improved rule-based logic with more nuanced criteria
        # High-risk factors:
        high_risk_count = 0
        
        # Age risk (higher age = higher risk)
        if age > 60:
            high_risk_count += 2
        elif age > 50:
            high_risk_count += 1
            
        # BMI risk (higher BMI = higher risk)
        if bmi > 35:  # Very high BMI
            high_risk_count += 2
        elif bmi > 30:  # High BMI
            high_risk_count += 1
            
        # Smoker risk (smoking = higher risk)
        if smoker == 1:
            high_risk_count += 3
            
        # Children risk (more children = more risk)
        if children > 3:
            high_risk_count += 2
        elif children > 1:
            high_risk_count += 1
            
        # Determine eligibility based on risk score
        # Maximum possible score: 2 (age) + 2 (bmi) + 3 (smoker) + 2 (children) = 9
        # Decline insurance for extremely high risk (score >= 7)
        if high_risk_count >= 7:
            print(f"Rule-based prediction: Not eligible (risk score: {high_risk_count})")
            return 0  # Not eligible
        else:
            print(f"Rule-based prediction: Eligible (risk score: {high_risk_count})")
            return 1  # Eligible
    
    except Exception as e:
        print(f"Critical error in make_prediction: {str(e)}")
        traceback.print_exc()
        # Return a conservative prediction in case of error
        return 0  # Default to not eligible in case of errors

# Function to predict insurance cost level
def predict_cost_level(age, bmi, children, smoker):
    """Predict cost level with more granularity"""
    # Calculate a risk score for cost
    cost_points = 0
    
    # Age factor (older = higher cost)
    if age > 60:
        cost_points += 25
    elif age > 50:
        cost_points += 20
    elif age > 40:
        cost_points += 15
    elif age > 30:
        cost_points += 10
    else:
        cost_points += 5
        
    # BMI factor (higher BMI = higher cost)
    if bmi > 35:
        cost_points += 25
    elif bmi > 30:
        cost_points += 20
    elif bmi > 25:
        cost_points += 15
    else:
        cost_points += 5
        
    # Children factor (more children = higher cost)
    cost_points += children * 5
    
    # Smoker factor (smoking dramatically increases cost)
    if smoker == 1:
        cost_points += 40
    
    # Determine cost level based on points
    if cost_points >= 70:
        return "Very High Cost"
    elif cost_points >= 50:
        return "High Cost"
    elif cost_points >= 30:
        return "Medium Cost"
    else:
        return "Low Cost"

# Route for Home & Prediction
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    insurance_decision = None
    cost_prediction = None
    prediction_details = None
    
    if request.method == "POST":
        try:
            # Debug code to log all form values
            print("Form data received:")
            for key, value in request.form.items():
                print(f"  {key}: {value}")
            
            # Get user input
            age = int(request.form["age"])
            bmi = float(request.form["bmi"])
            children = int(request.form["children"])
            smoker = 1 if request.form["smoker"] == "yes" else 0
            
            # Check if the user provided region or income
            region = request.form.get("region", "")
            income = request.form.get("income", "")
            
            # Process input based on what the user provided
            feature_name = ""
            if income and income.strip():
                try:
                    income = float(income)
                    user_data = np.array([[age, bmi, children, smoker, income]])
                    feature_name = "income"
                except ValueError:
                    print(f"Invalid income value: {income}")
                    raise ValueError(f"Invalid income value: '{income}'")
            elif region and region.strip():
                # Only use region if it's provided and not empty
                if region in regions:
                    region_encoded = regions.index(region)
                    user_data = np.array([[age, bmi, children, smoker, region_encoded]])
                    feature_name = "region"
                else:
                    print(f"Invalid region: {region}")
                    raise ValueError(f"Invalid region: '{region}'")
            else:
                # If neither region nor income is provided, default to the first region
                user_data = np.array([[age, bmi, children, smoker, 0]])
                feature_name = "region (default)"
            
            print(f"Processed user data: {user_data}")
            
            # Make eligibility prediction
            prediction = make_prediction(user_data)
            print(f"Prediction result: {prediction}")
            
            # Determine insurance decision and cost
            if prediction == 1:
                insurance_decision = "Eligible for Insurance"
                # Use improved cost prediction function
                cost_prediction = predict_cost_level(age, bmi, children, smoker)
                
                # Add prediction details
                risk_factors = []
                if age > 50:
                    risk_factors.append("Age over 50")
                if bmi > 30:
                    risk_factors.append("BMI above 30")
                if smoker == 1:
                    risk_factors.append("Smoker")
                if children > 2:
                    risk_factors.append("Multiple dependents")
                
                if risk_factors:
                    prediction_details = "Risk factors: " + ", ".join(risk_factors)
                else:
                    prediction_details = "Low risk profile"
            else:
                insurance_decision = "Not Eligible for Insurance"
                cost_prediction = "N/A"
                
                # Add rejection details
                rejection_factors = []
                if age > 60:
                    rejection_factors.append("Age above threshold")
                if bmi > 35:
                    rejection_factors.append("BMI above threshold")
                if smoker == 1:
                    rejection_factors.append("Smoking status")
                if children > 3:
                    rejection_factors.append("Number of dependents")
                
                if rejection_factors:
                    prediction_details = "Rejection factors: " + ", ".join(rejection_factors)
                else:
                    prediction_details = "Combined risk factors too high"
                
            # Debug print for verification
            print(f"Final decision: {insurance_decision}, Cost: {cost_prediction}")

        except Exception as e:
            print(f"Error in prediction processing: {str(e)}")
            traceback.print_exc()
            insurance_decision = "Error in Prediction"
            cost_prediction = "N/A"
            prediction_details = f"Error: {str(e)}"

    return render_template("index.html", 
                           insurance_decision=insurance_decision,
                           cost_prediction=cost_prediction,
                           prediction_details=prediction_details,
                           regions=regions, 
                           corr_matrix=corr_matrix_path, 
                           scatter_plot=scatter_plot_path,
                           age_chart=age_chart_path,
                           smoker_chart=smoker_chart_path)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)