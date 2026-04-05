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





"""
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

plt.show()"""


print("WITHOUT FEATURE ENGINERRING")