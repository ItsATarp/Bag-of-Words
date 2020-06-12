from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
#from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review,features="lxml").get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

if __name__ == '__main__':
    #reading training and test data sets
    train = pd.read_csv('/Users/kevinxu/Desktop/ML_project/labeledTrainData.tsv', sep = '\t') 
    test = pd.read_csv('/Users/kevinxu/Desktop/ML_project/testData.tsv', sep = '\t') 

    #nltk.download()

    # Initialize an empty list to hold the clean train and test reviews
    clean_train_reviews = []
    clean_test_reviews = []

    print ("Cleaning and parsing the training set and test set movie reviews...\n")
    for i in range(0, len(train["review"])):
       clean_train_reviews.append(" ".join(review_to_wordlist(train["review"][i], True)))
       clean_test_reviews.append(" ".join(review_to_wordlist(test["review"][i], True)))
    
    x = clean_train_reviews     # training x has stop words removed
    y = train["sentiment"]
    test_x = clean_test_reviews # testing x has stop words removed

    vectorizer = TfidfVectorizer(stop_words='english')  
    x = vectorizer.fit_transform(x)         # create word vector for training words
    test_x = vectorizer.transform(test_x)   # create word vector for test words
    x.toarray()
 
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.1, random_state=50)
    print ("Training model using random forest..")
    
    # Initialize a Random Forest classifier
    forest = RandomForestClassifier()
    forest = forest.fit(x_train,y_train)
    predictions = forest.predict(test_x)
    submission_file = pd.DataFrame(data={"id":test["id"], "sentiment":predictions})
    submission_file.to_csv('/Users/kevinxu/Desktop/ML_project/submission.csv', index=False)