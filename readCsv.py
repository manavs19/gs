import csv
with open('Initial_Training_Data.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		print ', '.join(row)