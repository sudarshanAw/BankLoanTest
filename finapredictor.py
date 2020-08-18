#importing libraries
import pandas as pnd

from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import joblib


#importing dataset
dataset = pnd.read_csv('data/bankloan.csv')
#creating a new dataset and removing all the na
dataset_final = dataset.dropna()
#checking for existing na's
dataset_final.isna().any()
#dropping load_id as it is of no use
dataset_final = dataset_final.drop('Loan_ID', axis=1)

#loan amount is given in different unit need to multiply by 1000
dataset_final['LoanAmount'] = (dataset_final['LoanAmount'] * 1000).astype(int)

#Counting No's and Yes in Loan Status
Counter(dataset_final['Loan_Status'])
#Counting YES
Counter(dataset_final['Loan_Status'])['Y']
#Counting NO
Counter(dataset_final['Loan_Status'])['N']
#getting percentage of yes
Counter(dataset_final['Loan_Status'])['Y']/dataset_final['Loan_Status'].size

dataset_final.columns

#getting dependent and independent variables and encoding them
pre_X = dataset_final.drop('Loan_Status',axis=1)
pre_y = dataset_final['Loan_Status']
#encoding the X
X_dummies = pnd.get_dummies(pre_X)
#encoding the Y using map
y_dummies = pre_y.map(dict(Y = 1, N = 0))

#As the dataset is imbalanced using smote
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
smote = SMOTE()
X1,y = smote.fit_sample(X_dummies,y_dummies)
#scaling the data
sc_X = MinMaxScaler()
X_scaled =  sc_X.fit_transform(X1)

#splitting the train test data
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size=0.2, random_state=42, shuffle=True)

#creating a neural networks
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#layer 1
classifier.add(Dense(400,activation='relu',kernel_initializer='random_normal',input_dim= X_train.shape[1] ))
#hidden layer 2
classifier.add(Dense(400,activation='relu',kernel_initializer='random_normal'))
classifier.add(Dense(10,activation='relu',kernel_initializer='random_normal'))
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train,y_train,batch_size=20,epochs=50, verbose=0)

#evaluation of model
eval_model = classifier.evaluate(X_train, y_train)
eval_model

#predicting the output
y_pred = classifier.predict(X_test)
#as the final function was sigmoid the output is between 0 and 1
#now deciding how to classify between 0 and 1
y_pred = (y_pred > 0.5)

#accuracy and visualization
from  sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)

file = 'loan_model.pkl'
joblib.dump(classifier,file)

file1 = 'scalar.pkl'
joblib.dump(sc_X,file1)

file2 = 'ohecolumns.pkl'
joblib.dump(X_dummies.columns,file2)