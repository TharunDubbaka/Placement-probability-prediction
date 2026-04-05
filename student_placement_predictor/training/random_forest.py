import pandas as pd
import sklearn as skl 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import classification_report
import seaborn as sns 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE #Using to oversample minority class which is the low one
from sklearn.linear_model import LogisticRegression




df = pd.read_csv("data//placement_chance_prediction_dataset.csv")

mapping={"Low":0, "Medium":1, "High":2}
df["Placement_Chance"]=df["Placement_Chance"].map(mapping)
df["Placement_Chance"] = df["Placement_Chance"].replace({0:1})
df["Skill_score"]=(df["Coding_Hours_Per_Week"]*0.4+df["Aptitude_Test_Score"]*0.4+df["Communication_Skills_Rating"]*10*0.2)
df["Experience_Score"]=(df["Projects_Completed"]+df["Internships"]+df["Hackathon_Participation"])


#print(df["Placement_Chance"].value_counts())  O/P was 2 - 894, 1 - 299, 0 - 7 (The 0 class has only 7 values), we should take care of that
#print(df.head())
#print(df.shape)
#print(df.head())



X=df.drop("Placement_Chance",axis=1)
y=df["Placement_Chance"]
#print(X.head())
#print(y.head())

bsmote=df["Placement_Chance"].value_counts()


smote=SMOTE(random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42, stratify=y) #WE added stratify=y to deal with imbalance classes
X_trainresample,y_trainresample=smote.fit_resample(X_train,y_train)
asmote=y_trainresample.value_counts()

rfs=RandomForestClassifier(n_estimators=100, max_depth=10,random_state=42, class_weight="balanced")
#THis is with smote
rfs.fit(X_trainresample,y_trainresample)
rfspreds=rfs.predict(X_test)
rfsprobs=rfs.predict_proba(X_test)
#print(preds)
rfsacc=accuracy_score(y_test,rfspreds)
rfscr=classification_report(y_test,rfspreds)
print(" RF SMOTE, ACCURACY AND CLASSIFICATOIN REPORT: ",rfsacc,rfscr)

rfns=RandomForestClassifier(n_estimators=150, max_depth=10,random_state=42,class_weight="balanced")
rfns.fit(X_train,y_train)
rfnspreds=rfns.predict(X_test)
rfnsprobs=rfns.predict_proba(X_test)
#print(preds)
rfaccns=accuracy_score(y_test,rfnspreds)
rfcrns=classification_report(y_test,rfnspreds)
print(" RF NO SMOTE ACCURACY, PRECISION:",rfaccns,rfcrns)

lgs=LogisticRegression(max_iter=1000,class_weight="balanced")
lgs.fit(X_trainresample,y_trainresample)
predslgs=lgs.predict(X_test)
print("ACCURACY AND CR OF SMOTE: ", accuracy_score(y_test,predslgs),classification_report(y_test,predslgs))

lgns=LogisticRegression(max_iter=1000,class_weight="balanced")
lgns.fit(X_train,y_train)
predslgns=lgns.predict(X_test)
print("ACCURACY AND CR OF NO SMOTE: ", accuracy_score(y_test,predslgns),classification_report(y_test,predslgns))

rfmodel=rfns
joblib.dump(rfmodel,"models//RF_MODEL.pkl")

lgmodel=lgns
joblib.dump(lgmodel,"models//lgmodel.pkl")






from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


y_prob = rfns.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=2)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1])  # diagonal line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()



print("WITH FEATURE ENGINERERING")

"""
RF SMOTE, ACCURACY AND CLASSIFICATOIN REPORT:  0.9333333333333333               precision    recall  f1-score   support

           1       0.83      0.93      0.88        61
           2       0.98      0.93      0.95       179

    accuracy                           0.93       240
   macro avg       0.90      0.93      0.92       240
weighted avg       0.94      0.93      0.93       240

 RF NO SMOTE ACCURACY, PRECISION: 0.9583333333333334               precision    recall  f1-score   support

           1       0.92      0.92      0.92        61
           2       0.97      0.97      0.97       179

    accuracy                           0.96       240
   macro avg       0.95      0.95      0.95       240
weighted avg       0.96      0.96      0.96       240

ACCURACY AND CR OF SMOTE:  1.0               precision    recall  f1-score   support

           1       1.00      1.00      1.00        61
           2       1.00      1.00      1.00       179

    accuracy                           1.00       240
   macro avg       1.00      1.00      1.00       240
weighted avg       1.00      1.00      1.00       240

ACCURACY AND CR OF NO SMOTE:  0.9916666666666667               precision    recall  f1-score   support

           1       0.97      1.00      0.98        61
           2       1.00      0.99      0.99       179

    accuracy                           0.99       240
   macro avg       0.98      0.99      0.99       240
weighted avg       0.99      0.99      0.99       240

WITH FEATURE ENGINERERING


 RF SMOTE, ACCURACY AND CLASSIFICATOIN REPORT:  0.95               precision    recall  f1-score   support

           1       0.92      0.89      0.90        61
           2       0.96      0.97      0.97       179

    accuracy                           0.95       240
   macro avg       0.94      0.93      0.93       240
weighted avg       0.95      0.95      0.95       240

 RF NO SMOTE ACCURACY, PRECISION: 0.9416666666666667               precision    recall  f1-score   support

           1       0.96      0.80      0.88        61
           2       0.94      0.99      0.96       179

    accuracy                           0.94       240
   macro avg       0.95      0.90      0.92       240
weighted avg       0.94      0.94      0.94       240

ACCURACY AND CR OF SMOTE:  0.9958333333333333               precision    recall  f1-score   support

           1       1.00      0.98      0.99        61
           2       0.99      1.00      1.00       179

    accuracy                           1.00       240
   macro avg       1.00      0.99      0.99       240
weighted avg       1.00      1.00      1.00       240

ACCURACY AND CR OF NO SMOTE:  0.9916666666666667               precision    recall  f1-score   support

           1       0.97      1.00      0.98        61
           2       1.00      0.99      0.99       179

    accuracy                           0.99       240
   macro avg       0.98      0.99      0.99       240
weighted avg       0.99      0.99      0.99       240

WITHOUT FEATURE ENGINERRING"""


#NO smote has better recall and f1 score which means we should consider No smote for our final. 
# Feature enginerring has imporved our model's recall from 0.8 to 0.92 for class 1










#Observations and analysis
# corr_matrix = df.corr(numeric_only=True)
# target_corr = corr_matrix['Placement_Chance'].sort_values(ascending=False)
# print(target_corr)

# I pasted the same code in a new python file, but i used feature enginerring. This has shown that 

#print(df.shape) I have noticed that the precision, recall and F1 score for class 0 is 0.00 and support is 1. This means class imbalance is there
#To overcome that, I tried SMOTE which is an oversampling technique but it didnt really do much, so I combined the "low" and medium classes. The values came upto somethign like 390 for merged and 717 for the high. 
#Still imbalanced, so I replaced smote after merging the two classes. Now it works well with a precision of around 90% and recall around 100, which definitely shows our dataset is easily seperable or there might be a data leakage.  
"""plt.figure()
sns.heatmap(df.corr(),annot=True,fmt=".2f")
plt.title("Feature coorelation")"""# This plot showed that feature engineered scores have a good coorelation with our placement chance.
"""
sns.boxplot(x="Placement_Chance",y="CGPA",data=df)
plt.title("CGPA vs Placement") #This plot showed that more CGPA could have impact on our chance"""

"""
importance=pd.DataFrame({"Feature":X.columns,"Importance":rfns.feature_importances_}).sort_values(by="Importance",ascending=False)
plt.figure()
plt.barh(importance["Feature"],importance["Importance"])
plt.title("Feature importance")
plt.gca().invert_yaxis()# This plot showed that CGPA does matter in making decision

plt.figure()
plt.subplot(1,2,1)
bsmote.plot(kind="bar")
plt.title("Before")

plt.subplot(1,2,2)
asmote.plot(kind="bar")
plt.title("After") #the handling is successfull


sns.countplot(x=y_train)
plt.title("Class distribution")
plt.show() # THe same but uses sns for plotting
"""