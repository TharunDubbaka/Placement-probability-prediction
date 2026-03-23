# Placement-probability-prediction
A machine learning project that predicts a student's placement potential using academic performance, skills, and experience. The system estimates the probability of placement and compares predictions from multiple models.


This project analyzes student data such as CGPA, coding practice, internships, and aptitude scores to predict placement outcomes.

Instead of directly predicting "placed/not placed", the model estimates placement likelihood using a combination of classification probabilities.

## Key Features
Predicts placement potential (Medium / High)
- Calculates estimated placement probability
- Uses 2 ML models:
    - Logistic Regression
    - Random Forest
- Handles class imbalance and feature engineering
- Provides interpretable output with probabilities
- Compares model behavior (confidence vs realism)
## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Joblib (model persistence)
## Feature Engineering
- Skill Score
- Combines coding hours, aptitude, and communication
- Experience Score
- Based on projects, internships, and hackathons
## Models Used
### Logistic Regression
Linear model
Produces highly confident predictions
Useful for understanding feature relationships
### Random Forest
Ensemble model
More robust and realistic probabilities
Captures non-linear patterns
### Placement Probability Logic

Since the dataset does not include a direct "placed/not placed" label:

Placement Probability = P(High) + 0.5 × P(Medium)
High → strong placement likelihood
Medium → partial likelihood

This provides a more realistic estimate of placement chances.

### Example Output
Random Forest:
Medium: 76.93%
High: 23.07%
Estimated Placement Probability: 61.5%

Logistic Regression:
High: 99.54%
Estimated Placement Probability: 99.54%


## Key Learnings
Handling class imbalance (SMOTE vs class merging)
Understanding model confidence vs real probability
Difference between linear and ensemble models
Importance of feature engineering
Interpreting ML outputs for real-world use

## Note

This is an educational project. The placement probability is an estimated metric derived from model outputs and not a real-world guarantee.

## Future Improvements
Add real placement (0/1) dataset
Deploy using Streamlit web app
Apply probability calibration
Add more advanced models (XGBoost, Neural Networks)
## How to Run
pip install -r requirements.txt
python test.py
## Why This Project Stands Out
Goes beyond simple prediction → focuses on probability interpretation
Shows model comparison and reasoning
Demonstrates real-world thinking in ML design
