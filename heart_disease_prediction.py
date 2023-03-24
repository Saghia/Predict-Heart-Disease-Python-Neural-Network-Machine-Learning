import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("heart_research.csv")

# CORRELAZIONI TRA I DATI
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(16,16))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
#pic1
sns.barplot(x=dataset["sex"],y=dataset["target"])
plt.show()
#pic2
sns.barplot(x=dataset["cp"],y=dataset["target"])
plt.show()
#pic3
sns.barplot(x=dataset["restecg"],y=dataset["target"])
plt.show()
#pic4
sns.barplot(x=dataset["exang"],y=dataset["target"])
plt.show()
#pic5
sns.barplot(x=dataset["slope"],y=dataset["target"])
plt.show()
#pic6
sns.barplot(x=dataset["fbs"],y=dataset["target"])
plt.show()
#pic7
sns.barplot(x=dataset["ca"],y=dataset["target"])
plt.show()
#pic8
sns.barplot(x=dataset.query('thal != 0')["thal"], y=dataset["target"])
plt.show()
#pic9
sns.distplot(dataset.query('thal != 0')["thal"])
plt.show()

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
# I Valori 11,1 e 2000 sono importanti, grazie a essi raggiungiamo un accuracy del 85%
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=2000)

Y_pred_nn = model.predict(X_test)

print() # --- TEST GENERICO SU UN PAZIENTE ---
print("--- IL PAZIENTE CON QUESTI VALORI : ")
print(X_test.iloc[0])
print(f"--- HA DATO QUESTO RISULTATO (0 Assente - 1 Presente) : {Y_pred_nn[0]}")
print(f"--- IL VALORE REALE DEL PAZIENTE E' (0 Assente - 1 Presente) : {Y_test.iloc[0]} ")
a = float(Y_pred_nn[0])
b = 1-a
valore = b

if Y_test.iloc[0]==0 and a<0.500000 :
    print("--- LA PREDIZIONE E' STATA : CORRETTA")

rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
print()
score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)
print("--- La PRECISIONE DELLA PREDIZIONE GENERALE E' STATA DEL : "+str(score_nn)+" %")