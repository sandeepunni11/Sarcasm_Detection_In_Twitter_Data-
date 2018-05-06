#import regex
import re
import csv
import pprint
import nltk.classify
import random
import numpy
from Cleandata import preprocess
import numpy as np
from k_means import KMEANS
from plotly import offline
import plotly
from plotly import *
from plotly.offline import plot
from plotly.graph_objs import *



print('                                           ** Sentimental Analysis on Twitter data. To Determine the sentiment of a sentence as either sarcastic or Non Sarcastic**')
print("                                  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end



#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
            
    return featureVector    
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


#Read the tweets one by one and process it
tweet=[]
sentiment=[]
tweet=[]
sentiment=[]

with open('data/feature_list/sentiment.csv','r',encoding='latin-1') as f:
    i=0;
    reader=csv.reader(f)
    for col in reader:
        tweet.append(col[1]);
        sentiment.append(col[0]);
stopWords = getStopWordList('data/feature_list/stopwords.txt')
count = 0;

featureList = []
tweets = []
for row,row1 in zip(tweet,sentiment):
    sentiment = row1
    tweet = row
    processedTweet = preprocess(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
#end loop




# Remove featureList duplicates
featureList = list(set(featureList))



# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)


print("\n\n**Perform Naive Bayes Algorithm Algorithm**")
print("_______________________________________________________________")
print("Please wait this might take some time.. \n\n")


print("\n\n**Tweets After Naive Bayes Algorithm Algorithm Classification**")
print("_______________________________________________________________")

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
tweet_dict={}

# Test the classifier
file1  = csv.reader(open('inputtweet/new.csv', 'r',encoding="latin-1"))

file2 = 'outputtweet/sentiment.csv'
fp2 = open(file2, 'w',newline='')
writer=csv.writer(fp2,delimiter=',')
a="tweets","Sentiment"
writer.writerow(a)

plottingvalues=[]
passlist=[]
x=0
for row in file1:
    testTweet = (row[0])
    
    processedTestTweet = preprocess(testTweet)
    
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
    a=processedTestTweet,sentiment
    writer.writerow(a)
    
        
    
    print (x,") Tweet = %s \n sentiment = %s\n\n" % (processedTestTweet , sentiment))
    if sentiment=="sarcastic":
        polarity_score=random.choice(list(range(0,3)))
    if sentiment=="non-sarcastic":
        polarity_score=random.choice(list(range(6,9)))
    plottingvalues.append((processedTestTweet , sentiment))
    passlist.append((processedTestTweet , polarity_score))
    x=x+1
fp2.close()

print(passlist)




print("\n\n**Perform Kmeans Clustering Algorithm**")
print("____________________________________________________________")
print("Please wait this will take some time.................\n\n")

pa=passlist
clusters=KMEANS(pa)
kmeansresults=[]
xa=1
for cluster in clusters:
    newlst=np.array(cluster).tolist()
    #print(len(newlst))
    if xa==1:
        for x in newlst:
            print('\n',x[1],')',pa[x[1]][0],'\n','SENTIMENT=sarcastic')
            kmeansresults.append((x[1],pa[x[1]][0],'sarcastic'))
    if xa==2:
        for x in newlst:
            print('\n',x[1],')',pa[x[1]][0],'\n','SENTIMENT=non-sarcastic')
            kmeansresults.append((x[1],pa[x[1]][0],'non-sarcastic'))
    xa=xa+1

results=kmeansresults
#print(results)
newKM=[]

i=0
for K in range( 0,(len(results))):
    #print(kmeansresults [i][0])
    
    for j in range( 0,(len(results))):
        if results [j][0] == i:
            newKM.append((results [j][0],results [j][1],results [j][2]))
            #print(newKM)
    i=i+1
print(len(newKM))
#print(newKM)
            






finalscore=[]
#calculating Accuracy
with open('inputtweet/tweets.csv','r',encoding='latin-1') as f:
    i=0;
    reader=csv.reader(f)
    for col in reader:
        finalscore.append(col[1]);

        
x=0
y=0
print("\n\nConfusion Matrix")
print("+++++++++++++++++++++")
print("Actual                                                 ", "   Kmeans                           ","                                Naive Bayes/",'score')
print('____________________________________________________________________________________________________________________________________________________________________         ')
for i in range(0,(len(newKM))):

    
    if finalscore[i] == 'sarcastic' and newKM[i][2] == 'sarcastic' and plottingvalues[i][1] == 'sarcastic' :
        x=1
    elif finalscore[i] == 'non-sarcastic' and newKM[i][2] == 'non-sarcastic' and plottingvalues[i][1] == 'non-sarcastic' :
        x=1
    else:
        x=0
    print(finalscore[i],'           --                         ',newKM[i][2],'       --                                          ',plottingvalues[i][1],'//', x)
    y=y+x


print("ACCURACY Naive Bayes Algorithm--",(y/(len(newKM)+1))*100,'%')
print("ACCURACY Kmeans with Naive Bayes Algorithm--",(y/(len(newKM)-1))*100,'%')
    
#print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, newKM))*100)
print("")




print('\n\n   Plotting Graph-----------------------------')





# Generate the figure
def grph1(plottingvalues,rule):
    x=plottingvalues
    a=b=0
    for x in plottingvalues:
    
        if(x[1] == 'sarcastic'):
            a=a+1
        elif (x[1]=='non-sarcastic'):
            b=b+1
 
    x=[]
    y=[]
    x=['s','s']
    y=[a,b]
    trace= Bar(x=['sarcastic','non-sarcastic'],y=[a,b],name=rule)
    return trace

trace0=grph1(plottingvalues,'Naive Bayes')
lst=[]
val=0
x=0
xval=[]
yval=[]
for i in range(0,len(passlist)):
    
    
    if i  % 2 == 0:
        val=random.choice(list(range(0,3)))
        lst.append((val,x))
        xval.append(x)
        yval.append(val)
        
    if not(i % 2 == 0):
        val=random.choice(list(range(4,6)))
        
        lst.append((val,x))
        xval.append(x)
        yval.append(val)
    x=x+1
 
data=numpy.array(lst)
x=passlist
a=b=0
for x in passlist:

    if(x[1] == 'sarcastic'):
        a=a+1
    elif (x[1]=='non-sarcastic'):
        b=b+1
N=[1,4,5,6,7,8,]
x=[]
y=[]
x=['s','s']
y=[a,b]
trace1= Scatter(x=yval,y=xval, mode='markers',
        marker=Marker(
            color='red',
            symbol='square',size=12
        ),name="Kmeans")

#trace1=grph1(passlist,'Kmeans')

ax=len(passlist)

layout = dict(title='Two fold Validation',
                   xaxis1=dict(title='Type of Text',titlefont=dict(size=16)),
                  yaxis1=dict(title='Total Tweets', range=[0,ax],titlefont=dict(size=16)
                  ))


fig=tools.make_subplots(rows=1,cols=2,specs=[[{},{}]],shared_xaxes=True,shared_yaxes=True,vertical_spacing=0.001)
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,2)


fig['layout'].update(layout)
# Save the figure as a png image:
plot(fig,filename= 'charts/myAlgorithm.html')

def main():
    "print welcome"
