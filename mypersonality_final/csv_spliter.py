import csv

clean_file = "mypersonality_clean.csv"
binary_file = "mypersonality_binary.csv"
test_file = "mypersonality_test.csv"
fields = []
rows = []

#Uses a clean file as the reader, shouldn't be altered
with open(clean_file, 'rt', encoding='utf-8', errors='ignore') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
    print("Total no. of rows: %d"%(csvreader.line_num))

#Create the new training file from 99/100 lines from the clean file
with open(binary_file,'wt', newline='', errors='ignore') as binarycsvfile:
    csvwriter = csv.writer(binarycsvfile)
    csvwriter.writerow(fields)
    for num, row in enumerate(rows):
        if (num % 100 != 0):
            csvwriter.writerow(row)

#Create the new test file from 100 lines from the clean file
with open(test_file,'wt', newline='', errors='ignore') as testcsvfile:
    csvwriter = csv.writer(testcsvfile)
    csvwriter.writerow(fields)
    for num, row in enumerate(rows):
        if (num % 100 == 0):
            csvwriter.writerow(row)
