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
df = pd.read_csv("placement_chance_prediction_dataset.csv")
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

rfs=RandomForestClassifier(n_estimators=150, max_depth=10,random_state=42, class_weight="balanced")
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
joblib.dump(rfmodel,"RF_MODEL.pkl")

lgmodel=lgns
joblib.dump(lgmodel,"lgmodel.pkl")























#Observations and analysis

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