import os
import numpy as np

# kmeans clustering algorithm
# data = set of data points
# k = number of clusters
# c = initial list of centroids (if provided)
#
def cmp(l1,l2):
    val=0
    for i,j in zip(l1,l2):
        #print(i,j)
        if i==j:
            print(i,j)
            val=0
        else:
            val=1
    return val
def kmeans(data, k, c):
    data1=data
    centroids = []

    centroids = randomize_centroids(data, centroids, k)  

    old_centroids = [[] for i in range(k)] 

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(data, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster[0], axis=0).tolist()
            index += 1

    
    print("\n\n++ The total number of data instances is: " + str(len(data)))
    print("\n++ The total number of iterations necessary is: " + str(iterations))
    print("\n++ The means of each cluster are: " + str(centroids))
    print("\n\n**The clusters are as follows:")
    for cluster in clusters:
        print("Cluster with a size of " + str(len(cluster)) + " starts here:")
        print(np.array(cluster).tolist())
        print("Cluster ends here.\n\n")
        
    
    
    return clusters

        
# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.      
def euclidean_dist(data, centroids, clusters):
    for instance in data:  
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min([(i[0], np.linalg.norm(instance[0]-centroids[i[0]])) \
                            for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append((instance[0],instance[1]))
        except KeyError:
            clusters[mu_index] = [instance[0],instance[1]]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.        
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters


# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids


# check if clusters have converged    
def has_converged(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 1000
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

import numpy
import random
def KMEANS(passlist):
    lst=[]
    val=0
    x=0
    
    for i in range(0,len(passlist)):
        
        lst.append((passlist[i][1],x))
        x=x+1
    
    data=numpy.array(lst)
    #print(data)
    clusters=kmeans(data, 2,2)
    return clusters
##










##pa= [('i love g able to sleep its the next g', 0), ('buying subscribers for shame', 0), ('gosh i love drama lol ', 1), ('ouch did that hu ', 1), ('when i die i want my grave to offer free wifi so that people visit more often ', 0), ("i just love it when my husband has the volume full blast on his laptop while i' g to watch tv", 2), ('lucky you said the unlucky one', 1), ('waking up for work at 4am is g fun ', 1), ('wow i wish i could spend every thursday night at the bar faithfully lol responsibilities', 0), ('re haha very funny i still love you', 0), ("but let's not worry about the vets and the american g makes sense please tell me what sense this makes", 0), ('some people just need a hug around the neck with a rope :p ', 2), (' munity school refugees fund: donate somalia school ', 6), ("oh wonderful it's 4 am and i'm awake oh good ", 2), ('name the toxic nature and failure of our education system d ', 7), ('a film to spire all people', 6), ('how big data is # g education all across globe', 6), ('how to help more college students graduate ', 8), (' pany to work for @wipro', 1), ("we're excited to talk design and today ah not", 1), ('please watch - the future of g is here /4546665433', 0), ('mother languages ponents of quality ternational motherlanguageday', 6), ('oh right i forgot legs take 100 years to grow back ', 0), ("i'm just shitty g on the phone but if i'm g person than it's you ", 1), ('yay baseball for the next 47 months ', 1), ('media festival at apeejay stya university: media festival was a platform for lively debat htcampus ', 6), ("i love when people who mon core mon core educate yourself 'aholes", 1), ('india aims to reshape education with novation ', 8), ('5 days left to register ceu rmt massagetherapy toronto ', 7), (' the the country ', 1), ('well what cidence even president told the g to me on the phone', 2), ("here's a strategy: kill them all ;) ", 2), ('we want trump for pope ', 1), ('business passion business toptags name g entrepreneurship grind hustle learn star ', 7), ('keep up the good work name literacymatters ', 0), ('really dude is this all you got i expected a lot more from you', 1), ('when i die make sure they play lonely day by system of a down as i to the ground the crematorium ', 2), ('re time to take your to the next level  good luck ', 0), ('i love when i get to work with someone who refuses to see t of view but their own not woh the effo im out ', 2), ('delhi corp co hiring careerarc can mend anyone for this job ', 8), ('can mend anyone for this job springfield mo hiring careerarc ', 8), ('big surprise( ) old white guy leads among a bloomberg politics poll', 2), ('haha life is great when you are around', 0), ("too much g service / expansion of gov't union enterprise evidence of pre-k value shaky pabudget", 6), ("and you sir are g example of 'academic excellence' ", 1), ('your gym shout outs are hilarious ', 2), ("good job name name your app works sooo awesome i'm sooo happy with it ", 1), ('reagan was 80 years ago sarcasm', 0), ('the toxic nature and failure of our education system d ', 8), ('time to do some g on my day off yay ', 2)]
##
##
##
##
##clusters=KMEANS(pa)
##kmeansresults=[]
##xa=1
##for cluster in clusters:
##    newlst=np.array(cluster).tolist()
##    if xa==1:
##        for x in newlst:
##            #print(x[1],pa[x[1]][0],'sarcastic')
##            kmeansresults.append((x[1],pa[x[1]][0],'sarcastic'))
##    if xa==2:
##        for x in newlst:
##            #print(x[1],pa[x[1]][0],'non-sarcastic')
##            kmeansresults.append((x[1],pa[x[1]][0],'non-sarcastic'))
##    xa=xa+1
##i=0
##(kmeansresults)
##new=[]
##print(len(kmeansresults))
##for K in range( 0,(len(kmeansresults)-1)):
##    #print(kmeansresults [i][0])
##    i=0
##    for j in range( 0,(len(kmeansresults)-1)):
##        if kmeansresults [K][0] == i:
##            new.append((kmeansresults [K][0],kmeansresults [K][1],kmeansresults [K][2]))
##        i=i+1
###print(new)
##
##newKM=[]
##
##i=0
##for K in range( 0,(len(kmeansresults))):
##    #print(kmeansresults [i][0])
##    
##    for j in range( 0,(len(kmeansresults))):
##        if kmeansresults [j][0] == i:
##            newKM.append((kmeansresults [j][0],kmeansresults [j][1],kmeansresults [j][2]))
##    i=i+1
##print(len(newKM))
###print(newKM)
##            
##
