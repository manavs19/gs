import csv
import time
import datetime
from sklearn import svm, cross_validation, tree
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest, chi2

trainingFileName = "Initial_Training_Data.csv"
testFileName = "Initial_Test_Data.csv"

data = []
data_map = {}

classes = []
classes_map = {}

with open(trainingFileName, 'rb') as csvfile:
	reader = csv.reader(csvfile)
	reader.next()
	for line in reader:
		row = []
		numCols = len(line)
		for i in range(1, numCols-1):
			if i==8 or i==12:
				continue
			if not line[i]:#empty
				row.append(float('nan'))
				continue

			if i==7:#days_to_settle
				row.append(int(line[i]))
				continue

			if str(line[i]) not in data_map:
				data_map[str(line[i])] = len(data_map)
			row.append(data_map[str(line[i])])
		data.append(row)

		currClass = str(line[numCols-1])
		if currClass not in classes_map:
			classes_map[currClass] = len(classes_map)
		classes.append(classes_map[currClass])

imp = Imputer()
imp = imp.fit(data)
data = imp.transform(data)

print data.shape
featureSelector= SelectKBest(chi2, k=6) #k=4 : 31.88%
featureSelector.fit(data, classes)
# print featureSelector.get_support()
data = featureSelector.transform(data)
print data.shape

# clf = tree.DecisionTreeClassifier()

clf=svm.SVC() # 56 %
scores = cross_validation.cross_val_score(clf, data, classes, cv=10)
print scores
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)


clf.fit(data,classes)
print "yo2"

test_data = []
isins = []
with open(testFileName, 'rb') as csvfile:
	reader = csv.reader(csvfile)
	reader.next()
	for line in reader:
		isins.append(str(line[0]))
		row = []
		numCols = len(line)
		
		for i in range(1, numCols):
			if i==8 or i==12:
				continue
			
			if not line[i]:#empty
				row.append(float('nan'))
				continue

			if i==7:#days_to_settle
				row.append(int(line[i]))
				continue

			if str(line[i]) not in data_map:
				data_map[str(line[i])] = len(data_map)
			row.append(data_map[str(line[i])])
		test_data.append(row)

imp = Imputer()
imp = imp.fit(test_data)
test_data = imp.transform(test_data)

test_data = featureSelector.transform(test_data)
print test_data.shape
print "yo3"

with open('output_file.csv', 'wb') as csvfile:
	sw = csv.writer(csvfile)	
	sw.writerow(['ISIN', 'Risk_Stripe'])
	for i in range(0, len(test_data)):
		dec = clf.predict(test_data[i])
		sw.writerow([str(isins[i]), 'Stripe ' + str(dec[0])])


