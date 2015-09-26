from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.naive_bayes import GaussianNB
import csv
import time
import datetime

def getTimestamp(s):
	""" Format:15-May-14 """
	return time.mktime(datetime.datetime.strptime(s, "%d-%b-%y").timetuple())

trainingFileName = "Initial_Training_Data.csv"
testFileName = "Initial_Test_Data.csv"

data = []
data_map = {}

classes = []
classes_map = {}

MEANDIFF=0

with open(trainingFileName, 'rb') as csvfile:
	reader = csv.reader(csvfile)
	reader.next()
	for line in reader:
		row = []
		numCols = len(line)
		startTime=0
		endTime=0

		if (not line[8] or not line[12]):
			continue
		if getTimestamp(line[8]) > getTimestamp(line[12]):
			continue
		for i in range(1, numCols-1):
			if not line[i]:#empty
				row.append(float('nan'))
				continue

			if i==7:#days_to_settle
				row.append(int(line[i]))
				continue

			if i==8:
				startTime = getTimestamp(line[i])
				continue

			if i==12:
				endTime = getTimestamp(line[i])
				continue

			if str(line[i]) not in data_map:
				data_map[str(line[i])] = len(data_map)
			row.append(data_map[str(line[i])])
		MEANDIFF=MEANDIFF + endTime - startTime
		row.append(endTime - startTime)
		data.append(row)

		currClass = str(line[numCols-1])
		if currClass not in classes_map:
			classes_map[currClass] = len(classes_map)
		classes.append(classes_map[currClass])

MEANDIFF = float(MEANDIFF) / len(data)
imp = Imputer()
imp = imp.fit(data)
data = imp.transform(data)

print data.shape
# for k in range(18,0,-1):
# data = SelectKBest(chi2, k=6).fit_transform(dd, classes)
featureSelector= svm.LinearSVC(C=0.01, penalty="l1", dual=False)
featureSelector.fit(data, classes)
# print featureSelector.get_support(), featureSelector.get_params()
data = featureSelector.transform(data)
print data.shape

print "yo1"

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
		startTime=0
		endTime=0
		if (not line[8] or not line[12]):
			line[8] = 1
			line[12] = 1+ MEANDIFF
		else:
			line[8] = getTimestamp(line[8])
			line[12] = getTimestamp(line[12])

		if line[8] > line[12]:
			line[8] = 1
			line[12] = 1+MEANDIFF

		for i in range(1, numCols):
			if not line[i]:#empty
				row.append(float('nan'))
				continue

			if i==7:#days_to_settle
				row.append(int(line[i]))
				continue

			if i==8:
				startTime = line[8]
				continue

			if i==12:
				endTime = line[12]
				continue

			if str(line[i]) not in data_map:
				data_map[str(line[i])] = len(data_map)
			row.append(data_map[str(line[i])])
		row.append(endTime-startTime)
		test_data.append(row)

# for r in test_data:
# 	print len(r)

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

