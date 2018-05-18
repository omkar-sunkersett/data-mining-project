# Author: Omkar

import csv

fread = open("/Users/omkarsunkersett/Downloads/inspections/chicago.csv", 'r')
rows = csv.reader(fread)
fwrite = open("/Users/omkarsunkersett/Downloads/inspections/chicago-final.csv", 'w')
fwrite.write('Facility Name,Facility Type,Risk Level,Address,Zip Code,Inspection Date,Type of Inspection,Result,Number of Violations,Violation Code 1,Violation Code 2,Violation Code 3,Violation Code 4,Violation Code 5,Violation Code 6,Violation Code 7,Violation Code 8,Violation Code 9,Violation Code 10,Violation Code 11,Violation Code 12,Violation Code 13,Violation Code 14,Violation Code 15,Violation Code 16,Violation Code 17,Violation Code 18,Violation Code 19,Violation Code 20,Violation Code 21,Violation Code 22,Violation Code 23\n')
rows.next()

for row in rows:
	row[13] = row[13].strip()
	if row[13] != '':
		if ' | ' not in row[13]:
			vcodes = row[13].split(' - Comments: ')[0].split('.')[0]
		else:
			 vcodes = ','.join([s.split()[0].rstrip('.') for s in row[13].split(' | ')])
	else:
		vcodes = ''
	fwrite.write(row[1].replace(',','')+','+row[4].replace(',','')+','+row[5].replace(',','')+','+row[6].replace(',','')+','+row[9].replace(',','')+','+row[10].replace(',','')+','+row[11].replace(',','')+','+row[12].replace(',','')+','+vcodes+'\n')
fwrite.close()
fread.close()

fread = open("/Users/omkarsunkersett/Downloads/inspections/nyc.csv", 'r')
rows = csv.reader(fread)
fwrite = open("/Users/omkarsunkersett/Downloads/inspections/nyc-final.csv", 'w')
fwrite.write('Facility Name,Borough,Street Address,Zip Code,Cuisine,Inspection Date,Violation Code,Criticality,Type of Inspection\n')
rows.next()

for row in rows:
	fwrite.write(row[1].replace(',','')+','+row[2].replace(',','')+','+row[3].replace(',','')+" "+row[4].replace(',','')+','+row[5].replace(',','')+','+row[7].replace(',','')+','+row[8].replace(',','')+','+row[10].replace(',','')+','+row[12].replace(',','')+','+row[17].replace(',','')+'\n')
fwrite.close()
fread.close()
