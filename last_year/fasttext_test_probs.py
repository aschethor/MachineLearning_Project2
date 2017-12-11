import csv

def create_csv_submission(ids, y_pred, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            if r2!=[]:
                writer.writerow({'Id':int(r1),'Prediction':int(r2[0])})
            else: #Tweets which were not classified, will have a positive value
                writer.writerow({'Id':int(r1),'Prediction':int(1)}) 


def create_csv_probs(ids, y_pred, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            if r2!=[]:
                writer.writerow({'Id':int(r1),'Prediction':(r2)}) #add int and subscript
            else: #Tweets which were not classified, will have a positive value
                writer.writerow({'Id':int(r1),'Prediction':int(100)}) 

def create_csv_revision(ids, y_pred, tweet, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction', 'Tweet']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2, r3 in zip(ids, y_pred,tweet):
            if r2!=[]:
                writer.writerow({'Id':int(r1),'Prediction':(r2), 'Tweet':str(r3)}) #add int and subscript [0]
            else: #Tweets which were not classified, will have a positive value
                writer.writerow({'Id':int(r1),'Prediction':int(1), 'Tweet':str(r3)})
            

###############################################################

import fasttext
import numpy as np
import random


train_data_dir='data/mixed_pos_neg_labeled_FT.txt'
#train_data_dir='data/mixed_pos_neg_FT.txt'
#train_data_dir='results/mixed_pos_neg.txt'
train_data_dir_train='results/mixed_pos_neg_train.txt'
train_data_dir_test='results/mixed_pos_neg_test.txt'


with open(train_data_dir) as d:
	all_tweets = d.readlines()

#tweets = np.array(all_tweets)

N = len(all_tweets)
ratio_train = 0.8 #Or 0.9? (0.86280)
ind = random.sample(range(N),N)
ind_train=ind[1:int(N*ratio_train)+1]
ind_test = ind[int(N*ratio_train)+1:] # How to fix this???
new_tweets = [all_tweets[e] for e in ind_train]
test_tweets =[all_tweets[e] for e in ind_test]
print(len(all_tweets))
print(len(new_tweets))
print(len(test_tweets))

with open(train_data_dir_train, "wb") as f:
	for item in new_tweets:	
		 f.write(bytes(item,'UTF-8'))

with open(train_data_dir_test, "wb") as f:
	for item in test_tweets:	
		 f.write(bytes(item,'UTF-8'))


#0.84760
#classifier = fasttext.supervised(train_data_dir, 'results/model',epoch=20, lr=0.005,loss='ns', ws=5,min_count=3,word_ngrams=3,thread=4,bucket=2000000)
#classifier = fasttext.supervised(train_data_dir_train, 'results/model',epoch=5, lr=0.005,loss='ns', ws=5,min_count=3,word_ngrams=3,thread=4,bucket=2000000) 0.852
classifier = fasttext.supervised(train_data_dir_train, 'results/model',epoch=5, lr=0.022,loss='ns', ws=5,min_count=4,word_ngrams=3,thread=4,bucket=2000000)
#Preparing the test dataset

fin=open('data/test_data.txt','r')
#fin=open('data/test_data_pre_LR_FT.txt','r')
#fin=open('results/test_data_pre.txt','r')
test_data=fin.read().splitlines()

print("Testing model...\n")

#results=classifier.test('results/mixed_pos_neg.txt')
results=classifier.test(train_data_dir_test)
print("Precision: %f" % results.precision)
print("Recall: %f" % results.recall)

#Getting the predictions
print("Getting the predictions...\n")
pred_labels=classifier.predict(test_data) ############## Modified by Camila ########
pred_p=classifier.predict_proba(test_data)

###Counting how many predictions the classifier made
print("Counting how many predictions the classifier made:\n")
count_pos=0
count_neg=0
count_no_pred=0
list_no_pred=[]

for i in range(len(pred_labels)):
    if pred_labels[i]==['1']:
        count_pos+=1
    if pred_labels[i]==['-1']:
        count_neg+=1
    if pred_labels[i]==[]:
        count_no_pred+=1
        list_no_pred.append(i)

print("Positive tweets: %d" % count_pos)
print("Negative tweets: %d" % count_neg)
print("Total of predictions: %d\n" %(count_pos+count_neg))
print("Tweets without prediction: " )

if count_no_pred>0:
	for i in range(len(list_no_pred)):
		print("%d: %s" %(list_no_pred[i]+1,test_data[list_no_pred[i]]))

#print(pred_labels.shape)

#pred_prod = [a[1] if int(a[0])==-1 else 1-a[1]  for x in pred_labels for a in x]
#pred_label =  [a[0] for x in pred_labels for a in x ]

pred_prods = np.zeros((len(pred_labels),1))

print(pred_prods.shape)
for ind, x in enumerate(pred_p):
	for a in x:
		if int(a[0])==-1:
			pred_prods[ind]=a[1]
		elif int(a[0])==1:
			pred_prods[ind]=1-a[1]
		else:
			pred_prods[ind]=0
		
pred_prods=pred_prods.ravel().tolist()
###########################
#Generating lists for id
id_labels=[]
for i in range(len(pred_labels)):
    id_labels.append(i+1)

print("Generating file for submission")
OUTPUT_PATH_submission = 'results/FastText_output_submission.csv'
OUTPUT_PATH_probs = 'results/FastText_output_probs.csv'
OUTPUT_PATH_revision = 'results/FastText_output_revision.csv'
#For submission
create_csv_submission(id_labels, pred_labels, OUTPUT_PATH_submission)
#For revision
create_csv_revision(id_labels, pred_labels, test_data, OUTPUT_PATH_revision)
#For probs
create_csv_probs(id_labels, pred_prods, OUTPUT_PATH_probs)
print("Done")
