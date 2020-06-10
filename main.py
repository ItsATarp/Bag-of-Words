from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk

if __name__ == '__main__':
    print ("hello")
   # train = pd.read_csv('/Users/kevinxu/Desktop/ML_project/labeledTrainData.tsv', 'data', 'labeledTrainData.tsv'), header=0, \
   #                 delimiter="\t", quoting=3)
    #test = pd.read_csv('/Users/kevinxu/Desktop/ML_project/testData.tsv', 'data', 'testData.tsv'), header=0, delimiter="\t", \
    #               quoting=3 )
    train = pd.read_csv('/Users/kevinxu/Desktop/ML_project/labeledTrainData.tsv', sep = '\t') 
    test = pd.read_csv('/Users/kevinxu/Desktop/ML_project/testData.tsv', sep = '\t') 

   
    #print ('Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...')
    
    #### YOU NEED TO UNCOMMENT THIS AND DOWNLOAD IT ONCE #####
    ####### nltk.download()  # Download text data sets, including stop words ######

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    clean_test_reviews = []

    print ("Cleaning and parsing the training set and test set movie reviews...\n")
    for i in range(0, len(train["review"])):
       clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
       clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))
    print(train["review"][0])
    print(clean_train_reviews[0])
    print(clean_test_reviews[0])
    x = clean_train_reviews
    y = train["sentiment"]

    test_x = clean_test_reviews
    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(x)
    test_x = vectorizer.transform(test_x)
    #vectorizer = TfidfVectorize(stop_words='english', max_features=10000)
    x.toarray()
 
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.1, random_state=50)
    print ("Training the random forest (this may take a while)...")
    
    # Initialize a Random Forest classifier
    forest = RandomForestClassifier()
    forest = forest.fit(x_train,y_train)
    y_val_pred = forest.predict(x_val)
    predictions = forest.predict(test_x)
    submission_file = pd.DataFrame(data={"id":test["id"], "sentiment":predictions})
    submission_file.to_csv('/Users/kevinxu/Desktop/ML_project/submission.csv', index=False)