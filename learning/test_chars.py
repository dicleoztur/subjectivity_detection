'''
Created on Sep 24, 2015

@author: dicle
'''

import pandas as pd
import numpy as np
from nltk.util import ngrams
from nltk import ConditionalFreqDist
import os

from txtprocessor import texter
from corpus import extractnewsmetadata, metacorpus
from sentimentfinding import IOtools



# textmap = {docid : content}, n is the gram unit
# outputs a cfd
def subword_char_ngram_counts2(textmap, n=2):

    allngramunits = []
    for textid, text in textmap.iteritems():
        ngramunits = []
        words = texter.getwords(text, nopunctuation=False, nostopwords=False)
        for w in words:
            ngramunits.extend((textid, ngrams(w, n)))
        allngramunits.extend(ngramunits)
    
    cfd = ConditionalFreqDist(allngramunits)
    return cfd




def subword_char_ngram_df(fileids, normalize, n=2):
    cfd = get_subword_char_ngram_cfd(fileids, n) 
    return cfd_to_df(cfd, normalize)

def superword_char_ngram_df(fileids, normalize, n=2):
    cfd = get_superword_char_ngram_cfd(fileids, n)
    return cfd_to_df(cfd, normalize)




# textmap = {docid : content}, n is the gram unit
# outputs a cfd
def get_subword_char_ngram_cfd(fileids, n=2):

    allngramunits = []
    
    for fileid in fileids:
        filepath = extractnewsmetadata.newsid_to_filepath(fileid)
        _, text = extractnewsmetadata.get_news_article(filepath)

        ngramunits = []
        words = texter.getwords(text, nopunctuation=False, nostopwords=False)
        for w in words:
            ngramunits.extend(ngrams(w, n))
        
        for ngramunit in ngramunits:
            allngramunits.append((fileid, ngramunit))
    
    cfd = ConditionalFreqDist(allngramunits)
    return cfd


def get_superword_char_ngram_cfd(fileids, n=2):

    allngramunits = []
    
    for fileid in fileids:
        filepath = extractnewsmetadata.newsid_to_filepath(fileid)
        _, text = extractnewsmetadata.get_news_article(filepath)

        ngramunits = ngrams(text, n)
        
        for ngramunit in ngramunits:
            allngramunits.append((fileid, ngramunit))
    
    cfd = ConditionalFreqDist(allngramunits)
    return cfd



# cfd : { instanceid : (feature, count)}
# extracts a matrix m of size len(instanceids) x len(features) and 
#       m[i,j] = numoftimes the feature j occurs in instance i
def cfd_to_df(cfd, normalize=True):
    rownames = cfd.conditions()  # instance ids
    colnames = [] # feature ids
    for instanceid in rownames:
        colnames.extend(list(cfd[instanceid]))
    
    matrix = np.empty([len(rownames), len(colnames)], dtype=object)
    for i,instanceid in enumerate(rownames):
        for j,feature in enumerate(colnames):
            matrix[i][j] = cfd[instanceid][feature]
    if normalize:
        matrix = simple_normalize_ngram_matrix(matrix)
        
    df = pd.DataFrame(matrix, index=rownames, columns=colnames)
    return df


# each column is normalized by its total value (there are better methods as well)
def simple_normalize_ngram_matrix(countmatrix):
    matrix = countmatrix.copy()
    nr, _ = matrix.shape
    for i in range(nr):
        matrix[i, :] = np.divide(matrix[i, :], float(sum(matrix[i, :])))
    return matrix




if __name__ == '__main__':
    # sfjs
    #shgjshgjf
    
    
    print "hello"
    
    testoutput = "/home/dicle/Dicle/Tez/stylistic_comparison/test1/output/fullsets/"
    '''fileids = ['solhaber-economy-64909',  'solhaber-world-61859',  
               'radikal-turkey-1104771',  'radikal-economy-1126465',  
               'radikal-world-1107418',  'vakit-economy-307756',  
               'vakit-economy-320767',  'solhaber-politics-71804',  
               'radikal-world-1095301',  'vakit-turkey-261647']
    '''
    #fileids = ['solhaber-economy-64909']
    
    resultscsvpath = metacorpus.get_annotatedtexts_file_path(annotationtype="double", agreementype="fullagr")
    print resultscsvpath
    
    resultsdf = IOtools.readcsv(resultscsvpath, keepindex=False)
    print resultsdf.shape
    fileids = resultsdf["questionname"].tolist()
    
    
    for n in range(3, 5):
        subcharcsv = "subchar-" + str(n) + "gram.csv"
        supercharcsv ="superchar-" + str(n) + "gram.csv"
        
        dfsubchar = subword_char_ngram_df(fileids, normalize=True, n=n)
        psub = os.path.join(testoutput, subcharcsv)
        IOtools.tocsv(dfsubchar, psub, keepindex=True)
        del dfsubchar
        
        dfsuperchar = superword_char_ngram_df(fileids, normalize=True, n=n)
        psuper = os.path.join(testoutput, supercharcsv)
        #print dfsubchar.shape,"  ",dfsuperchar.shape
        IOtools.tocsv(dfsuperchar, psuper, keepindex=True)
    
    
    '''
    dfsubchar = subword_char_ngram_df(fileids, normalize=True, n=2)
    psub = os.path.join(testoutput, "subchar-bigram_10docs.csv")
    IOtools.tocsv(dfsubchar, psub, keepindex=True)
    del dfsubchar
    
    dfsuperchar = superword_char_ngram_df(fileids, normalize=True, n=2)
    psuper = os.path.join(testoutput, "superchar-bigram_10docs.csv")
    #print dfsubchar.shape,"  ",dfsuperchar.shape
    IOtools.tocsv(dfsuperchar, psuper, keepindex=True)
    '''
    
    