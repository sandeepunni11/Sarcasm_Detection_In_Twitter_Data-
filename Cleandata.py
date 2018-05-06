#preprocesses the tweets removes uwanted data 

import sys
import re 
#using regex operations

def  preprocess(Tweet):
    try:
        #Convert into lowercase
        Tweet = Tweet.lower()
        
        #Convert www.* or https?://* to URL
        Tweet = re.sub('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)',' ',Tweet)

        #Convert email to emails
        Tweet = re.sub('\w+@\w+\.(?:com|in)+',' ',Tweet)
        
        #Convert www.* to url
        Tweet = re.sub('\w+.\w+.(?:com|in)[///]+\w+\w+\s',' ',Tweet)
        Tweet = re.sub('\w+.\w+.(?:com|in)',' ',Tweet)

        #Convert @username to User
        Tweet = re.sub('@+\w+\s','',Tweet)

        #remove punctutaions
        Tweet = Tweet.replace('.','')
        Tweet = re.sub(',',' ',Tweet)
        Tweet = Tweet.replace('?','')
        Tweet = re.sub('!',' ',Tweet)
        
        #Remove additional white spaces
        Tweet = re.sub('[\s]+', ' ', Tweet)
        
        #Replace #word with word Handling hashtags
        Tweet = re.sub(r'#([^\s]+)', r'\1', Tweet)
        
        #trim
        Tweet = Tweet.strip('\'"')
        
        #Deleting happy and sad face emoticon from the tweet 
        a = ':)'
        b = ':('
        Tweet = Tweet.replace(a,'')
        Tweet = Tweet.replace(b,'')
        
        #Deleting the Twitter @username tag and reTweets
        tag = 'TWITTER_USER' 
        rt = 'rt'
        Tweet = Tweet.replace(tag,'')
        Tweet = Tweet.replace(rt,'')
        
        #trim
        Tweet = Tweet.strip('\'"')
        return Tweet
    except Exception as e:
        print(e)


#print(preprocess(" RT  @LOfan i'm the best there #is and you suck... !! www.f.com/go sandeep@gmail.com "))   
