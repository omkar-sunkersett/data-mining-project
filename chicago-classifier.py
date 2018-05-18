    import csv, time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn import svm
from sklearn.metrics import f1_score

start_time = time.time()
full_df = pd.read_csv('/Users/omkarsunkersett/Downloads/inspections/chicago-final.csv', keep_default_na=False)
full_df = full_df.query("Result == 'Pass' | Result == 'Fail'")
train_df, test_df = train_test_split(full_df, test_size = 0.7)
del full_df

facilities = dict()
train_data = []
train_labels = []
for index, row in train_df.iterrows():
	i = 1
	violations = []
	while i < 24:
		if row['Violation Code '+str(i)] != '':
			violations.append(row['Violation Code '+str(i)])
		i += 1
	violations = ' '.join(violations)
	if '1' in row['Risk Level']:
		risk_level = 'High'
	elif '2' in row['Risk Level']:
		risk_level = 'Medium'
	elif '3' in row['Risk Level']:
		risk_level = 'Low'
	else:
		risk_level = 'All'
	if row['Result'] == 'Fail':
		result = 0
	elif row['Result'] == 'Pass':
		result = 1
	if (row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code']) not in facilities.keys():
		facilities[(row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code'])] = {'n_pass': 0, 'n_fail': 0}
	if row['Result'] == 'Pass':
		facilities[(row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code'])]['n_pass'] += 1
	elif row['Result'] == 'Fail':
		facilities[(row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code'])]['n_fail'] += 1
	train_data.append(row['Facility Type']+' '+risk_level+' '+row['Type of Inspection']+' '+violations)
	train_labels.append(result) # 0: Fail, 1: Pass; Ground Truth
	
test_data = []
test_labels = []
for index, row in test_df.iterrows():
	i = 1
	violations = []
	while i < 24:
		if row['Violation Code '+str(i)] != '':
			violations.append(row['Violation Code '+str(i)])
		i += 1
	violations = ' '.join(violations)
	if '1' in row['Risk Level']:
		risk_level = 'High'
	elif '2' in row['Risk Level']:
		risk_level = 'Medium'
	elif '3' in row['Risk Level']:
		risk_level = 'Low'
	else:
		risk_level = 'All'
	if row['Result'] == 'Fail':
		result = 0
	elif row['Result'] == 'Pass':
		result = 1
	test_data.append(row['Facility Type']+' '+risk_level+' '+row['Type of Inspection']+' '+violations)
	test_labels.append(result) # 0: Fail, 1: Pass; Groud Truth

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

#classifier = BernoulliNB() # F1 Score:  0.912075797753 ; Elapsed Time:  67.49269104 seconds 
#classifier = MultinomialNB() # F1 Score:  0.879296472481 ; Elapsed Time:  77.1627590656 seconds
#classifier = RandomForestClassifier() # F1 Score:  0.960704837637 ; Elapsed Time:  72.624876976 seconds
#classifier = LogisticRegression() # F1 Score:  0.959769539464 ; Elapsed Time:  69.9529490471 seconds
#classifier = svm.SVC(kernel='rbf', probability=True) # F1 Score:  0.928968884009 ; Elapsed Time:  730.000464201 seconds
#classifier = svm.LinearSVC() # F1 Score:  0.959807806694 ; Elapsed Time:  70.702037096 seconds
classifier = SGDClassifier() # F1 Score:  0.961991430571 ; Elapsed Time:  69.5818319321 seconds
#classifier = Perceptron() # F1 Score:  0.946108704335 ; Elapsed Time:  76.9029090405 seconds

classifier.fit(train_vectors, train_labels)
prediction = classifier.predict(test_vectors)

i = 0
for index, row in test_df.iterrows():
	if (row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code']) not in facilities.keys():
		facilities[(row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code'])] = {'n_pass': 0, 'n_fail': 0}
	if prediction[i] == 1:
		facilities[(row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code'])]['n_pass'] += 1
	elif prediction[i] == 0:
		facilities[(row['Facility Name'], row['Facility Type'], row['Address'], row['Zip Code'])]['n_fail'] += 1
	i += 1

fwrite = open('/Users/omkarsunkersett/Downloads/inspections/chicago-results.csv', 'w')
fwrite.write('Facility Name,Facility Type,Address,Zip Code,Number of Inspections Passed, Number of Inspections Failed,Classifier Result\n')
for k,v in facilities.items():
	if v['n_pass'] >= v['n_fail']:
		fwrite.write(k[0]+','+k[1]+','+k[2]+','+k[3]+','+str(v['n_pass'])+','+str(v['n_fail'])+',Recommended\n')
	else:
		fwrite.write(k[0]+','+k[1]+','+k[2]+','+k[3]+','+str(v['n_pass'])+','+str(v['n_fail'])+',Not Recommended\n')
fwrite.close()

elapsed_time = time.time() - start_time

print "F1 Score: ",f1_score(test_labels, prediction, average='binary')
print "Elapsed Time: ",str(elapsed_time),"seconds"

