import joblib
import numpy as np

# =========================
# 1. LOAD MODEL
# =========================
lgmodel = joblib.load("lgmodel.pkl")
rfmodel = joblib.load("RF_MODEL.pkl")
print("Model loaded successfully!")

# =========================
# 2. INPUT
# =========================
"""should output high
cgpa = 8.5
internships = 2
coding_hours = 20
projects = 3
certifications = 1
communication = 4
hackathons = 2
aptitude = 80
#shoudl output negative
cgpa = 5.5
internships = 0
coding_hours = 3
projects = 0
certifications = 0
communication = 2
hackathons = 0
aptitude = 40
"""
cgpa = 7.0
internships = 1
coding_hours = 10
projects = 2
certifications = 1
communication = 3
hackathons = 1
aptitude = 65

skill_score = (coding_hours*0.4+aptitude*0.4+communication * 10 * 0.2)
experience_score = projects + internships + hackathons
sample = np.array([[cgpa, internships, coding_hours, projects,
                    certifications, communication, hackathons,
                    aptitude, skill_score, experience_score]])
lgprediction = lgmodel.predict(sample)   
lgprobability = lgmodel.predict_proba(sample)  
rfprediction = rfmodel.predict(sample)   
rfprobability = rfmodel.predict_proba(sample) 

print("RF_PROBS: ",rfprobability) 

rfmed = rfprobability[0][0]
rfhigh= rfprobability[0][1]

# =========================
# PLACEMENT PROBABILITY (RF)
# =========================
rfmed = rfprobability[0][0]
rfhigh = rfprobability[0][1]

rf_placement = rfhigh + 0.5 * rfmed

# =========================
# PLACEMENT PROBABILITY (LG)
# =========================
lgmed = lgprobability[0][0]
lghigh = lgprobability[0][1]

lg_placement = lghigh + 0.5 * lgmed

print("\n===== RANDOM FOREST =====")
print("Medium Probability:", round(rfmed * 100, 2), "%")
print("High Probability:", round(rfhigh * 100, 2), "%")
print("Estimated Placement Probability:", round(rf_placement * 100, 2), "%")

print("\n===== LOGISTIC REGRESSION =====")
print("Medium Probability:", round(lgmed * 100, 2), "%")
print("High Probability:", round(lghigh * 100, 2), "%")
print("Estimated Placement Probability:", round(lg_placement * 100, 2), "%")
def interpret(prob):
    if prob > 0.75:
        return "High chance of placement"
    elif prob > 0.5:
        return "Moderate chance of placement"
    else:
        return "Low chance of placement"

print("\nRF Interpretation:", interpret(rf_placement))
print("LR Interpretation:", interpret(lg_placement))



"""
print("RF")
if rfmed > rfhigh:
    print("Medium Chance")
    print("Medium Probability:", round(rfmed * 100, 2), "%")
else:
    print("Prediction: High")
    print("High Probability:", round(rfhigh * 100, 2), "%")
    
lgmed=lgprobability[0][0]
lghigh=lgprobability[0][1]
print("LG")
if lgmed > lghigh:
    print("Medium Chance")
    print("Medium Probability:", round(lgmed * 100, 2), "%")
else:
    print("Prediction: High")
    print("High Probability:", round(lghigh * 100, 2), "%")"""