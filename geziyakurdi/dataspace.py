'''
Created on Jul 6, 2013

@author: dicle
'''


import os
import nltk
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math

from txtprocessor import texter, listutils, dateutils
from sentimentfinding import IOtools
from sentimentfinding import keywordhandler
from languagetools import SAKsParser
from sentimentfinding import CFDhelpers
from sentimentfinding import plotter
from stats import classification
from sentimentfinding import Corpus, corpusfeatures



class DataSpace:
    corpus = Corpus()
    featurematrix = []
    doctermmatrix = []
   
    classlabels = []
    spacename = ""     # numoffiles_classtask
    
    def __init__(self):
        self.corpus = Corpus()
        self.featurematrix = []
        self.doctermmatrix = []
        
        self.classlabels = []
        self.spacename = ""
        
    def __getfeaturematrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_featurematrix.p"
        return self.featurematrix, fname
    def __getdoctermmatrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_doctermmatrix.p"
        return self.doctermmatrix, fname
    def __getcorpora(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_corpora.p"
        return self.corpora, fname
    
    
    def __dumpfeaturematrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_featurematrix.p"
        pickle.dump(self.featurematrix, open(fname, "wb"))
    def __dumpdoctermmatrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_doctermmatrix.p"
        pickle.dump(self.doctermmatrix, open(fname, "wb"))
    def __dumpcorpora(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_corpora.p"
        pickle.dump(self.corpora, open(fname, "wb"))
    
    
    ''' nfile is a dict storing the number of files per classlabel to be read  
        nfile default value 0 rec.'''
    def buildcorpus(self, nfile, resourcepath, classlabels, corpusname, taskname, plaintext, nostopwords):
        labelwisepathlist = {}
        
        for classlabel in classlabels:
            labelwisepathlist[classlabel] = []
            
        for classlabel in classlabels:
            p = resourcepath + os.sep + classlabel + os.sep
            fileids = []
            fileids = IOtools.getfilenames_of_dir(p, removeextension=False)[:nfile[classlabel]]
            
            labelwisepathlist[classlabel].extend(fileids)
            
            
        self.corpus.setname(corpusname)
        self.corpus.read_corpus(labelwisepathlist, plaintext, nostopwords)
        ncat = len(classlabels)
        self.spacename = taskname+"-"+str(nfile*ncat)+"texts"
        
        

    def compute_tfidf(self):
        
        
        
        ''' matrix leri duzelt.  csv olarak kaydet  '''
    def build_featurematrix(self):
        for corpus in self.corpora:
            datapoints = corpus.build_featurematrix()
            for k,v in datapoints.iteritems():
                self.featurematrix.append([k]+v+[corpus.label])
        self.record_matrix(self.featurematrix, "featureMATRIX")
       
        
    def build_termdocmatrix(self):
        cfdDocTerm = nltk.ConditionalFreqDist()
                
        #docs = []
        labelleddocs = []
        for corpus in self.corpora:
            cfd = corpus.build_termmatrix()
            label = corpus.label
            print label
            for term in cfd.conditions():
                #docs.extend(list(cfd[term]))
                #labelleddocs = [(doc, label) for doc in docs]
                #print list(cfd[term])
                for fileid in list(cfd[term]):
                    cfdDocTerm[term].inc(fileid)
                    labelleddocs.append((fileid, label))
        
        print labelleddocs           
        labelleddocs = list(set(labelleddocs))
        print labelleddocs
        CFDhelpers.recordCFD(cfdDocTerm, self.spacename+"CFDdocterm")
        
        matrix = []
        matrix.append(cfdDocTerm.conditions())
        
        
        for fileid,label in labelleddocs:
            row = []
            for term in cfdDocTerm.conditions():
                numofoccurrences = cfdDocTerm[term][fileid]
                row.append(numofoccurrences)
            self.doctermmatrix.append([fileid]+row+[label])
            matrix.append([fileid]+row+[label])
        
                       
        self.record_matrix(matrix, "DocTermMATRIXn")
        self.record_matrix(self.doctermmatrix, "DocTermMatrix")
        
        self.__dumpdoctermmatrix()
             
    
    def record_matrix(self, matrix, mname):
        fname = IOtools.matrixpath+os.sep+mname+"-"+self.spacename+"MATRIX.m"
        IOtools.todisc_matrix(matrix, fname)

#######    dataspace class end  ##########     
        




# pickle to make different plots
def buildcorpus(nfile, ncat, resourcename, path):
    resourcepath = path + os.sep + resourcename
    catnames = IOtools.getfoldernames_of_dir(resourcepath)[:ncat]
    
    featurematrix = []
    doctermmatrix = []
    cfdTermDoc = nltk.ConditionalFreqDist()
    
    for catname in catnames:
        fileids = []
        p = resourcepath + os.sep + catname + os.sep
        fileids.extend(IOtools.getfilenames_of_dir(p, removeextension=False)[:nfile])
        corpus = corpusfeatures.CorpusFeatures(fileids, resourcename+os.sep+catname, p)
        corpus.getfeatures()
        datapoints = corpus.build_featurematrix()
        for k,v in datapoints.iteritems():
            featurematrix.append([k]+v+[resourcename])
            
        corpus.plot_features()
        
        #doc term matrix
        cfd = corpus.build_termmatrix()
        for fileid in cfd.conditions():
            for term in list(cfd[fileid]):
                cfdTermDoc[fileid].inc(term)
    
    IOtools.todisc_matrix(featurematrix, IOtools.results_rootpath+os.sep+"MATRIX"+str(nfile*ncat)+"texts.txt", mode="a")
    #return featurematrix   

def plotcorpus(figname, xLabel, yLabel):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)       
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.title(figname)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 



def dataspace_prepare(nfile, taskname, rootpath, classlabels):
    dataspace = DataSpace()
    #rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/categorize/train/"
    #taskname = "3categorize" 
    dataspace.buildcorpus(nfile, rootpath, classlabels, taskname)
    #dataspace.build_termdocmatrix()
    dataspace.build_featurematrix()
    



def find_tfidf(doctermmatrix):        
    """ 
    MODIFY FROM http://blog.josephwilk.net/projects/latent-semantic-analysis-in-python.html
    tfidfmatrix is a numpy array
    Apply TermFrequency(tf)*inverseDocumentFrequency(idf) for each matrix element. 
        This evaluates how important a word is to a document in a corpus
           
        With a document-term matrix: matrix[x][y]
        tf[x][y] = frequency of term y in document x / frequency of all terms in document x
        idf[x][y] = log( abs(total number of documents in corpus) / abs(number of documents with term y)  )
        Note: This is not the only way to calculate tf*idf
    """


    numofdocs = doctermmatrix.shape[0]
    rows,cols = doctermmatrix.shape
    tfidfmatrix = np.array(doctermmatrix, copy=True)
   
    for row in xrange(0, rows): #For each document
       
        wordTotal= reduce(lambda x, y: x+y, tfidfmatrix[row] )

        for col in xrange(0,cols): #For each term
        
            #For consistency ensure all self.matrix values are floats
            tfidfmatrix[row][col] = float(tfidfmatrix[row][col])

            if tfidfmatrix[row][col]!=0:
                termDocumentOccurences = sum(tfidfmatrix[:,col])
                termFrequency = tfidfmatrix[row][col] / float(wordTotal)
                inverseDocumentFrequency = math.log(abs(numofdocs / float(termDocumentOccurences)))
                tfidfmatrix[row][col]=termFrequency*inverseDocumentFrequency

    return tfidfmatrix



def compute_TFIDF(corpuspath):
    return

''' doc1 and doc2 are lists of stemmed words 
    metric is the function that computes the similarity between two sets of words.
'''
def get_doc_similarity(doc1, doc2, metric):
    return metric(doc1, doc2)




if __name__ == "__main__":
    
    doc1 = ["dünya", "dikkat", "taksim", "meydan", "recep", "tayyip", "erdoğan", "hükümet", "karşı", "ol", "halk", "ayaklan", "çevril", "durum", "herkes", "zaman"]
    doc2 = ["ekonomik", "politika", "ol", "bağlılık", "karşı", "şimdiye", "kadar", "hükümet", "gelişim", "proje", "ad(i)", "alt", "istanbul", "kal", "küçük", "yeşil", "alan", "yok", "et", "niyet"]
    print nltk.metrics.jaccard_distance(set(doc1), set(doc2))
    print nltk.metrics.binary_distance(set(doc1), set(doc2))
    print nltk.metrics.masi_distance(set(doc1), set(doc2))