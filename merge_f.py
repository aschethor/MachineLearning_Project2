import os
import numpy as np
import csv

def create_csv_submission_ensemble(ids, y_pred, name):
	with open(name, 'w') as csvfile:
		fieldnames = ['Id','Prediction']
		writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator='\n')
		writer.writeheader()
		for r1,r2 in zip(ids, y_pred):
			if (r2>0.5): #Verificar con fasttext
				writer.writerow({'Id':int(r1),'Prediction':int(-1)})
			else:
				writer.writerow({'Id':int(r1),'Prediction':int(1)})


def create_csv_probs(ids, y_pred, name):
	with open(name, 'w') as csvfile:
		fieldnames = ['Id', 'Prediction']
		writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
		writer.writeheader()
		for r1, r2 in zip(ids, y_pred):
			writer.writerow({'Id':int(r1),'Prediction':(r2)}) #add int and subscript
			

'''
with open('results/logistic.csv', 'w') as csvfile:
	logistic = csvfile.readlines()
	
with open('results/fasttext.csv', 'w') as csvfile:
	fasttext = csvfile.readlines()
	
	
l = np.array(logistic)
f = np.array(fasttext)
'''
N = 200
f_total = np.zeros((10000,1)).ravel()

for i in range(N):
	print('Iteration',i+1)
	os.system('python3 fasttext_test_probs.py')
	f = np.genfromtxt('results/FastText_output_probs.csv', delimiter=",", skip_header=1)
	f_total = np.add(f[:,1],f_total)

pred_labels = f_total/N 



#Generating lists for id
id_labels=[]
for i in range(len(pred_labels)):
    id_labels.append(i+1)
	
	
print("Generating ensemble file for submission")
OUTPUT_PATH_submission = 'results/FastText_ensemble_submission.csv'
create_csv_submission_ensemble(id_labels, pred_labels, OUTPUT_PATH_submission)

print("Generating probs file")
OUTPUT_PATH_submission = 'results/fasttext.csv'
#pred_prods=pred_prods.ravel().tolist()
create_csv_probs(id_labels, pred_labels, OUTPUT_PATH_submission)
