# -*- coding: utf-8 -*- 
'''
Created on Sep 4, 2012

@author: dicle
'''

import os
import codecs
import re
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import math
import numpy as np
import pandas as pd

import dateutils, listutils
#from sentimentfinding import keywordhandler
from processing import Crawling2
from languagetools import bigramfinder

outputpath = "/home/dicle/Dicle/Tez/output/"

keyinputpath = "/home/dicle/Dropbox/Tez/system/keywordbase/"

stopwordsbase = keyinputpath+os.sep+"stopwords.txt"    #ilerde indislenebilir, stopwords folder kurup farkli dillerin listeleri ayri dosyalarda saklanir (31Mart). 






def readtxtfile(path):
    f = codecs.open(path,encoding='utf8')
    rawtext = f.read()
    f.close()
    return rawtext

'''
def readnewstext(path):
    wholetext = readtxtfile(path)
    
    txt = ""
    txtmarker1 = "<ttxtt>"
    txtmarker2 = "</ttxtt>"
    
    i1 = wholetext.find(txtmarker1)
    i2 = wholetext.find(txtmarker2)
    
    if i1 == -1:
        txt = ""
    else:
        txt = wholetext[i1 + len(txtmarker1) : i2]
        
    #print path[-10:]
    #print txt+"\n"
    
    return txt
'''

def getnewsmetadata(newspath, tags):
    wholetext = readtxtfile(newspath)
    tagdata = {}
   
    for tag in tags:
        marker1 = "<"+tag+">"
        marker2 = "</"+tag+">"
       
        data = Crawling2.extractitem(marker1, marker2, wholetext).lower()
        if tag == "date":
            data = dateutils.parsewrittendate(data)
        tagdata[tag] = data
    return tagdata


def readnewsitem(path):
    wholetext = readtxtfile(path)
    
    date = ""
    datemarker1 = "<date>"
    datemarker2 = "</date>"
    date = Crawling2.extractitem(datemarker1, datemarker2, wholetext).lower()
    date = dateutils.parsewrittendate(date)  
    
    txt = ""
    txtmarker1 = "<ttxtt>"
    txtmarker2 = "</ttxtt>"
    txt = Crawling2.extractitem(txtmarker1, txtmarker2, wholetext).strip()    # wholetext[markdate : ] is better
    
    return txt, date


''' reads the news item located at path, obtaining the whole text. then splits it by space, eliminates punctuation from each word and removes empty ones
    returns the list of words   '''
def getnewsitem(path, nostopwords=True):
    rawtext, date = readnewsitem(path)
    words = getwords(rawtext, nostopwords)
    words = filter(lambda x : len(x) > 0, words)
    return words, date



''' splits the input by space, eliminates punctuation from each word and removes empty ones. can also eliminate stopwords
    returns the list of words   '''
def getwords(rawtext, nopunctuation=True, nostopwords=True):
    words = rawtext.split()
    words = [word.strip() for word in words] # if (not word.isspace()) or (not len(word) == 0)]
    words = [word for word in words if (not word.isspace()) or (len(word) > 0)]
    words = filter(lambda x : len(x) > 0, words)
    if nopunctuation:
        words = eliminatepunctuation(words)
    
    if nostopwords:
        stopwords = readtextlines(stopwordsbase)
        '''stopwords = list(set(stopwords))
        stopwords.sort()'''
        words = [word for word in words if word not in stopwords]
    words = remove_endmarker(words, "(i)")
    words = remove_endmarker(words, "(ii)")
    return words


''' gen purpose reader   '''
def read_article(filepath, nostopwords=True):
    txt = readtxtfile(filepath)
        
    marker="Haziran 2013"
    mark = txt.find(marker)    # skip metadata
    txt = txt[mark+len(marker):]
    
    words = getwords(txt, nostopwords)
    return words



def readtextlines(path):
    f = codecs.open(path,"r", encoding='utf8')
    lines = f.readlines()
    lines = [line.strip() for line in lines if not line.isspace()]
    f.close()
    return lines

def splitToSentences(txt):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(txt.strip())
    #print '\n-----\n'.join(sents)
    return sents


# due to parser's returning ambiguous roots as root(i) or root(ii), 
# we eliminate those marks from the list as part of preprocessing.
def remove_endmarker(wordlist, mark):
    newlist = []
    for word in wordlist:
        if word.endswith(mark):
            word = word[: (-len(mark))]
        newlist.append(word)
    return newlist


def remove_endmarker_singleword(word, mark):
    newword = word
    if word.endswith(mark):
        newword = word[: (-len(mark))]
    return newword


# inputtext is a concatenated string of words, keywordlist is a list of words.
# returns true is inputtext contains at least one element of keywordlist
def keywords_search(words, keywordlist):
    
    for keyword in keywordlist:
        splitkeyword = keyword.split()
        windowsize = len(splitkeyword)
        i = 0
        contains = False
        
        while (contains is False) and (i <= (len(words)-windowsize)):
            searchingword = " ".join(words[i:i+windowsize])
            #print "%%%% ",searchingword, " && ",keyword
            if searchingword.strip() == keyword:
                print "%%%% ",searchingword, " && ",keyword
                contains = True
            i = i+1
        if contains is True:
            break
    return contains



'''
def eliminatepunctuation(wordlist):
    cleanwords = []    
    for w in wordlist:
        cleanword = ''.join(re.findall(r'\w+',w, flags=re.UNICODE))
        #cleanword = ''.join(re.findall(r'[(\w+) | (\w+\'\w+)]',w, flags=re.UNICODE))
        cleanwords.append(cleanword.lower())      # cleanwords.append(cleanword)
    return cleanwords
'''

def eliminatepunctuation(words, keepApostrophe=True):
    newwords = []
    for s in words:
        pattern=re.compile("[^\w']", flags=re.UNICODE)
        clean = pattern.sub('', s)
        if keepApostrophe:
            if clean.startswith("'") or clean.startswith("\""):
                clean = clean[1:]
            if clean.endswith("'") or clean.endswith("\""):
                clean = clean[:-1]
            
        newwords.append(clean.lower())
    return newwords


def eliminatepunctuation_singleword(word, keepApostrophe=True):
    pattern=re.compile("[^\w']", flags=re.UNICODE)
    clean = pattern.sub('', word)
    if keepApostrophe:
        if clean.startswith("'") or clean.startswith("\""):
            clean = clean[1:]
        if clean.endswith("'") or clean.endswith("\""):
            clean = clean[:-1]
    return clean
            
 
 
def cleanword(word):
    newword = word.encode('utf-8').lower().decode('utf-8')
    
    # puncutation eliminated
    newword = eliminatepunctuation_singleword(newword)  
    
    # markers from morphological analyser eliminated
    newword = remove_endmarker_singleword(newword, "(i)") 
    newword = remove_endmarker_singleword(newword, "(ii)")
    
    # numeric strings eliminated
    pattern = re.compile(r'^\d+$', flags=re.UNICODE)
    #print "clean word ",newword,"  ",type(newword)
    newword = pattern.sub('', newword)
        
    return newword  #.decode('utf-8').encode('utf-8')



####   text metrics    ####
def compute_tfidf_ondisc(freqdfpath, tfidfpath):
    doctermfreq = pd.read_csv(freqdfpath, index_col=0, sep="\t")
    numofdocs, numofwords = doctermfreq.shape
    docs = doctermfreq.index.values.tolist()
    terms = doctermfreq.columns.values.tolist()
    
    matrix = np.empty((numofdocs, numofwords))
    
    for i,doc in enumerate(docs):
        for j,term in enumerate(terms):
            tf = doctermfreq.iloc[i,j]
            df = np.count_nonzero(doctermfreq.iloc[:, j])
            
            idf = math.log(float(numofdocs) / df)
            matrix[i,j] = tf * idf
    
    matrix = np.around(matrix, decimals=4)
    
    doctermframe = pd.DataFrame(matrix, index = docs, columns=terms) 
    #doctermframe.to_csv(self.rootpath+os.sep+"matrix"+"doctermTFIDF.csv")
    doctermframe.to_csv(tfidfpath, sep="\t")
    return doctermframe



def compute_tfidf_online(doctermfreqdf):
    numofdocs, numofwords = doctermfreqdf.shape
    docs = doctermfreqdf.index.values.tolist()
    terms = doctermfreqdf.columns.values.tolist()
    
    matrix = np.empty((numofdocs, numofwords))
    
    for i,doc in enumerate(docs):
        for j,term in enumerate(terms):
            tf = doctermfreqdf.iloc[i,j]
            df = np.count_nonzero(doctermfreqdf.iloc[:, j])
            
            idf = math.log(float(numofdocs) / df)
            matrix[i,j] = tf * idf
    
    matrix = np.around(matrix, decimals=4)
    
    tfidfdf = pd.DataFrame(matrix, index = docs, columns=terms) 
    return tfidfdf



class CorpusReader:
    rpath = ""
    fileworddict = {}
    
    def __init__(self, path, N=5):
        self.rpath = path
        self.fileworddict = {}
        self.read_files(N)
    def read_files(self, N):
        fileids = os.listdir(self.rpath)[:N]
        for fileid in fileids:
            path = self.rpath + os.sep + fileid
            words, date = getnewsitem(path)[0]
            self.fileworddict[(fileid[:-4], date)] = words
    def getnumofiles(self):
        return len(self.fileworddict.keys())
    def getwordsoffile(self, fileid):
        return self.fileworddict[fileid]
    def getwordsofindex(self, ind):
        return self.fileworddict.values()[ind]
    def toscreen(self):
        for fileid,words in self.fileworddict.iteritems():
            print fileid," :  ",words[:10]




if __name__ == "__main__":
    
    
    s = u"aa"
    print s
    print cleanword(s)
    
    '''
    path = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/radikal/turkiye/"
    
    corpus5 = CorpusReader(path)
    print corpus5.toscreen()

    allbigrams = []
    allwords = []
    for i in range(corpus5.getnumofiles()):
        allwords.extend(corpus5.getwordsofindex(i))
        bigramlist = nltk.bigrams(corpus5.getwordsofindex(i))
        allbigrams.extend(bigramlist)
    

    finder = BigramCollocationFinder.from_words(allwords)
    #BigramAssocMeasures.
    
    for item in allbigrams:
        u, v = item
        print u," ",v
    
    
    freqbigram = nltk.FreqDist(allbigrams)
    for item in freqbigram.keys():
        u, v = item
        print "(",u,", ",v,")"," :  ",freqbigram[item]
        
    
    words = []
    bigramlist = nltk.bigrams(words)
    print bigramlist

    trigramlist = nltk.trigrams(words)
    print trigramlist
    
    print bigramfinder.bigram_bywordlist(words)
    '''
    
    '''
    countrypath = "/home/dicle/Dicle/Tez/dataset/country/"
    countries = ["turkey","syria"]
    
    keywords = readtextlines(countrypath+countries[0]+".txt")
    print keywords
    textpath = "/home/dicle/Dicle/Tez/dataset/inputtest/dispolitika.txt"
    text = readtxtfile(textpath)
    sents = splitToSentences(text)
    
    contains = False
    for sentence in sents:
       
        contains = keywords_search(sentence.lower().split(), keywords)
        if contains:
            print sentence
            
    '''        