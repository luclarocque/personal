import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt #plotting library
import seaborn as sns #statistical data visualization
import os
# %matplotlib inline


def curate_data(filename):

	train = pd.read_csv(filename)
	# train.info()
	# print(train.describe())
	# print(train.head(15).to_string())

	# # Check for missing/undefined values 
	# sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap="YlGnBu_r")
	# plt.show()

	train.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)
	# print(train.head(15).to_string())

	# #Plotting Survived against Sex
	# # sns.countplot(x='Survived',hue='Sex',palette='Set2',data=train)
	# sns.catplot(x="Sex", col="Survived",palette='Set2',data=train,kind="count")
	# plt.show()

	# #Plotting Survived against PClass
	# sns.catplot(x="Pclass", col="Survived",palette='Set2',data=train,kind="count")
	# plt.show()

	# #Plotting Survived against Embarked (location)
	# sns.catplot(x="Embarked", col="Survived",palette='Set2',data=train,kind="count")
	# plt.show()

	# plt.subplots(figsize=(9,5))
	# ax = sns.heatmap(train[["Survived","Pclass","Age","SibSp","Parch","Fare"]].corr(),annot=True, fmt = ".2f")
	# plt.show()
	# #From the correlation heatmap no variable seems to be highly correlated with another 
	# #so I won't have to drop any.

	#-----------

	# We must handle the missing Age inputs for some rows. Do some by ascribing
	#  to each missing age the average of the Pclass.

	meanAge1 = int(train[train['Pclass']==1]['Age'].mean())
	meanAge2 = int(train[train['Pclass']==2]['Age'].mean())
	meanAge3 = int(train[train['Pclass']==3]['Age'].mean())

	def impute_age(info):
		age = info[0]
		pclass = info[1]
		if pd.isnull(age):
			if pclass == 1:
				return meanAge1
			elif pclass == 2:
				return meanAge2
			else:
				return meanAge3
		else:
			return age

	train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

	# # Check for missing/undefined values 
	# sns.heatmap(train.isnull())
	# plt.show()

	# fix the few missing Embarked values using the most common value (mode)
	train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)

	# ----------

	# Feature scaling

	# # Checking the distribution of the fare variable
	# train['Fare'].hist(bins = 30,color="g")
	# sns.distplot(train["Fare"],bins = 30,color="g")
	# plt.show()

	def normalize(col):
		# return (col-col.median())/col.std()
		return col

	# Nromalize fare
	train['Fare'] = normalize(train['Fare'].map(lambda x: np.log(x) if x>0 else 0))
	# sns.distplot(train["Fare"],bins = 30)
	# plt.show()

	# Normalize Age
	train['Age'] = normalize(train['Age'])

	# ----------

	# Feature engineering

	# Dropping SibSp and Parch but creating a family feature with them.
	train['FamilySize'] = train['SibSp'] + train['Parch'] +1
	train.drop(['SibSp','Parch'],axis=1,inplace=True)

	# Extracting only title from name
	train_title = [s.split(',')[1].split('.')[0].strip() for s in train['Name']]
	train['Title'] = pd.DataFrame(train_title)
	train.drop('Name',axis=1,inplace=True)

	# Reducing number of titles
	train['Title'].replace('Mlle', 'Miss', inplace=True)
	train['Title'].replace('Ms', 'Miss', inplace=True)
	train['Title'].replace('Mme', 'Mrs', inplace=True)

	# Define the rest to be 'Rare'
	rare = train['Title'].unique()
	rare = list(filter(lambda x: x not in ['Miss','Mrs','Mr','Master'], rare))
	train['Title'].replace(rare, 'Rare', inplace=True)


	# title = pd.get_dummies(train['Title'],drop_first=True) # drop_first ('Master') since this is implied if rest are 0
	title = pd.get_dummies(train['Title'])

	train = pd.concat([train,title],axis=1)
	train.drop(['Title','Master'], axis=1, inplace=True)

	# create dummies for Embarked and Sex
	embarked = pd.get_dummies(train['Embarked'],drop_first=True)
	train = pd.concat([train,embarked],axis=1)
	train.drop('Embarked', axis=1, inplace=True)

	sex = pd.get_dummies(train['Sex'],drop_first=True)
	train = pd.concat([train,sex],axis=1)
	train.drop('Sex', axis=1, inplace=True)

	print('Data from', filename, 'curated. Sample data:')
	print(train.head(1).to_string(),'\n')
	return train


def build_model(data):

	train = data

	y = train['Survived']
	x = train.drop('Survived', axis=1)

	#Next we split the dataset into the train and test set
	#Test will be 30% of the data and the train will be 70%. By setting test_size = .3
	#This way we can test our models predictions on the test set to see how we did.
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


	# Find hyperparameters

	from sklearn.ensemble import RandomForestClassifier
	#--- use Grid Search for trial and error ---
	# from sklearn.model_selection import GridSearchCV

	# rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

	# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1,2,3,5], "min_samples_split" : [10,11,12,13], "n_estimators": [350, 400, 450, 500,550], "max_depth":[6,7,8,9]}

	# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

	# gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])

	# print(gs.best_score_)
	# print(gs.best_params_)
	# print(gs.scorer_)

	# output accuracy: 0.8462401795735129
	# using the following params:
	#  {'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 13, 'n_estimators': 350}

	# Define the model with these params
	rfmodel = RandomForestClassifier(random_state=0, n_estimators=350, criterion='gini', n_jobs=-1, max_depth = 8, min_samples_leaf=1, min_samples_split= 13)
	#Fitting the model to x_train and y_train
	rfmodel.fit(x_train,y_train)
	#Predicting the model on the x_test
	predictions = rfmodel.predict(x_test)

	# # Print precision, recall, f1-score
	# from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions), '\n')

	# --- Confusion Matrix ---
	# True positive (TP): Top left (139) are the one we predicted died and did die.
	# False positive (FP): Top right (14) are the ones we predicted they survive but they died
	# False negative (FN): Bottom left (42) are the ones we predicted they die but they survived
	# True negative (TN) :Bottom right (73) these are the ones we predicted would survive and they did
	# FP is also known as a "Type I error."
	# FN is also known as a "Type II error."
	from sklearn.metrics import  confusion_matrix
	print('Confusion matrix (True positive top left, True negative bottom right)')
	print(confusion_matrix(y_test,predictions),'\n')


	# # Permutation importance (I couldn't get this to work)
	# import eli5
	# from eli5.sklearn import PermutationImportance
	# perm = PermutationImportance(rfmodel, random_state=1).fit(x_test, y_test)
	# eli5.show_weights(perm, feature_names = x_test.columns.tolist())


	# #Applying K-Fold Cross Validation
	# from sklearn.model_selection import cross_val_score
	# accuracies = cross_val_score(estimator=rfmodel,X= x_train,y=y_train,cv=10)
	# print('10-fold cross validation')
	# print(accuracies) #Prints out the 10 different Cross Validation scores.

	return rfmodel



if __name__ == '__main__':
	traindata = curate_data('train.csv')
	model = build_model(traindata)
	testdata = curate_data('test.csv')
	predictions = model.predict(testdata)
	predictions = pd.DataFrame(predictions, columns=['Predicted'])
	testdata = pd.concat([testdata,predictions],axis=1)
	print(testdata.head(20).to_string())
	