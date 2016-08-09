'''
Created on Sep 18, 2015

@author: dicle
'''

import pandas as pd
import numpy as np
from nltk.util import ngrams
from nltk import ConditionalFreqDist

from txtprocessor import texter
from corpus import extractnewsmetadata


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
    colnames = list(set(colnames))
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


if __name__ == "__init__":
    # record two matrices
    # run on shell or inside this. record the results, interpret.
    
    print "hello"
    
    '''
    testoutput = "/home/dicle/Dicle/Tez/stylistic_comparison/test1/output/"
    fileids = ['solhaber-economy-64909',  'solhaber-world-61859',  
               'radikal-turkey-1104771',  'radikal-economy-1126465',  
               'radikal-world-1107418',  'vakit-economy-307756',  
               'vakit-economy-320767',  'solhaber-politics-71804',  
               'radikal-world-1095301',  'vakit-turkey-261647']
    
    dfsubchar = subword_char_ngram_df(fileids, normalize=True, n=2)
    dfsuperchar = superword_char_ngram_df(fileids, normalize=True, n=2)
    psub = os.path.join(testoutput, "subchar-bigram_10docs.csv")
    psuper = os.path.join(testoutput, "superchar-bigram_10docs.csv")
    
    
    print dfsubchar.shape,"  ",dfsuperchar.shape
    IOtools.tocsv(dfsubchar, psub, keepindex=True)
    IOtools.tocsv(dfsuperchar, psuper, keepindex=True)

    '''

