import os
import numpy as np
import pickle
import random

'''
Note: No obligation to use this code, though you may if you like.  Skeleton code is just a hint for people who are not familiar with text processing in python. 
It is not necessary to follow. 
'''


def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(None, symbols).strip(), lines)
    words = filter(None, words)
    return words
	
###############################################
######## YOUR CODE STARTS FROM HERE. ##########
###############################################

def shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    from sys import platform as _platform
    if _platform == 'darwin':
        pos_path = '/Users/josephpmcdonald/Dropbox/MachineLearning15/hw3/data/pos'
        neg_path = '/Users/josephpmcdonald/Dropbox/MachineLearning15/hw3/data/neg'
    else:
        pos_path = "/home/mcdonald/Dropbox/MachineLearning15/hw3/data/pos"
        neg_path = "/home/mcdonald/Dropbox/MachineLearning15/hw3/data/neg"
    
    
    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)
	
    review = pos_review + neg_review
    random.shuffle(review)
    return review
	
'''
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure. 
*Check it out. https://wiki.python.org/moin/UsingPickle
'''
 
if __name__ == "__main__":

    from collections import Counter

    reviews = shuffle_data()
    train_list = reviews[:1500]
    test_list = reviews[1500:]
    train = [(Counter(r[:-1]), r[-1]) for r in train_list]
    test = [(Counter(r[:-1]), r[-1]) for r in test_list]
    pickle.dump(train, open("train.pkl", "wb"))
    pickle.dump(test, open("test.pkl", "wb"))





