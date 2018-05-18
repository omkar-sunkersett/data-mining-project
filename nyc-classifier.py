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
full_df = pd.read_csv('/Users/omkarsunkersett/Downloads/inspections/nyc-final.csv', keep_default_na=False, encoding='latin-1', low_memory=False)
full_df = full_df.query("Criticality == 'Critical' | Criticality == 'Not Critical'")
train_df, test_df = train_test_split(full_df, test_size = 0.7)
del full_df

facilities = dict()
train_data = []
train_labels = []
for index, row in train_df.iterrows():
	if row['Criticality'] == 'Not Critical':
		result = 0
	elif row['Criticality'] == 'Critical':
		result = 1
	if (row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code']) not in facilities.keys():
		facilities[(row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code'])] = {'n_pass': 0, 'n_fail': 0}
	if row['Criticality'] == 'Not Critical':
		facilities[(row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code'])]['n_pass'] += 1
	elif row['Criticality'] == 'Critical':
		facilities[(row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code'])]['n_fail'] += 1
	train_data.append(row['Borough'].replace(' ','-')+' '+row['Cuisine'].replace(' ','-')+' '+row['Violation Code']+' '+row['Type of Inspection'].replace(' ','-'))
	train_labels.append(result) # 0: Not Critical, 1: Critical; Ground Truth

test_data = []
test_labels = []
for index, row in test_df.iterrows():
	if row['Criticality'] == 'Not Critical':
		result = 0
	elif row['Criticality'] == 'Critical':
		result = 1
	test_data.append(row['Borough'].replace(' ','-')+' '+row['Cuisine'].replace(' ','-')+' '+row['Violation Code']+' '+row['Type of Inspection'].replace(' ','-'))
	test_labels.append(result) # 0: Not Critical, 1: Critical; Ground Truth

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

#classifier = BernoulliNB() # F1 Score:  0.999993504342 ; Elapsed Time:  87.7441260815 seconds
#classifier = MultinomialNB() # F1 Score:  1.0 ; Elapsed Time:  71.885833025 seconds
#classifier = RandomForestClassifier() # F1 Score:  0.999974003185 ; Elapsed Time:  72.9299209118 seconds
#classifier = LogisticRegression() # F1 Score:  1.0 ; Elapsed Time:  74.9015779495 seconds
#classifier = svm.SVC(kernel='rbf', probability=True) # F1 Score:  0.99985088271; Elapsed Time:  3309.96746206 seconds
#classifier = svm.LinearSVC() # F1 Score:  1.0 ; Elapsed Time:  79.4777629375 seconds
classifier = SGDClassifier() # F1 Score:  1.0 ; Elapsed Time:  71.1480591297 seconds
#classifier = Perceptron() # F1 Score:  0.999987016107 ; Elapsed Time:  83.3481681347 seconds

classifier.fit(train_vectors, train_labels)
prediction = classifier.predict(test_vectors)

i = 0
for index, row in test_df.iterrows():
	if (row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code']) not in facilities.keys():
		facilities[(row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code'])] = {'n_pass': 0, 'n_fail': 0}
	if prediction[i] == 1:
		facilities[(row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code'])]['n_pass'] += 1
	elif prediction[i] == 0:
		facilities[(row['Facility Name'], row['Cuisine'], row['Street Address'], row['Zip Code'])]['n_fail'] += 1
	i += 1

results_df = []
for k,v in facilities.items():
	if v['n_pass'] >= v['n_fail']:
		results_df.append([k[0], k[1], k[2], k[3], str(v['n_pass']), str(v['n_fail']), 'Recommended'])
	else:
		results_df.append([k[0], k[1], k[2], k[3], str(v['n_pass']), str(v['n_fail']), 'Not Recommended'])

results_df = pd.DataFrame(results_df, columns=['Facility Name', 'Cuisine', 'Address', 'Zip Code', 'Number of Inspections Passed', 'Number of Inspections Failed', 'Classifier Result'])
results_df.to_csv('/Users/omkarsunkersett/Downloads/inspections/nyc-results.csv', encoding='latin-1', index=False)

elapsed_time = time.time() - start_time

print "F1 Score: ",f1_score(test_labels, prediction, average='binary')
print "Elapsed Time: ",str(elapsed_time),"seconds"

