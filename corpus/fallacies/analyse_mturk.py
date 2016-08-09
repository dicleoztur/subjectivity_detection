'''
Created on Sep 10, 2015

@author: dicle
'''

import os
import pandas as pd
import numpy as np
from collections import Counter

from sentimentfinding import IOtools

ANSWER_PREFIX = "Answer.s"
comment_suffix = "_comment"
nonsense_suffix = "_nonsense"
ANSWERCELL_SEP = " : "
mvote_colname = "majorityvote_"
cats = {"cat1" : ["MA0", "MA1", "MA2"], "cat2" : ["MA3", "MA4", "MA5"]}
EQUAL_VOTES = "EQ"

def get_answer_cols(df, remove_comments=True):
    
    allcols = df.columns.tolist()
    allcols = [str(c) for c in allcols]
    #print type(str(allcols[0]))
    answercols = []
    for colname in allcols:
        if colname.startswith(ANSWER_PREFIX):
            answercols.append(colname)
    
    if remove_comments:
        answercols = [col for col in answercols if not col.endswith(comment_suffix)]
    
    return answercols


def get_filenames(df, remove_comments=True):
    
    answercols = get_answer_cols(df, remove_comments)
    filenames = ["_".join(i.split("_")[:-1]) for i in answercols]
    filenames = list(set(filenames))
    return filenames



def get_annotator_answer_map(df):
    answercols = get_answer_cols(df)


def split_to_hits(df, outfolder):
    answercols = get_answer_cols(df)
    necessary_cols = ["assignmentid", "workerid"] + answercols
    
    hits = list(set(df["hitid"].tolist()))  # get unique hits
    
    hitcount = 0
    for hit in hits:
        print "processing ",hitcount," ",hit," ",
        hitdf = df[df["hitid"] == hit]
        hitdf = hitdf[necessary_cols]
        hitdf = hitdf.dropna(axis=1, how="all")
        print " dim: ",hitdf.shape
        hitcsvpath = os.path.join(outfolder, hit + ".csv")
        IOtools.tocsv(hitdf, hitcsvpath, keepindex=False)


# remove the sentenceid preceding the annotation
def strip_answers(df):
    indices = df.index.tolist()
    cols = df.columns.tolist()
    for i in indices:
        for j in cols:
            cell = df.loc[i, j]
            if "_" in cell:
                df.loc[i, j] = cell.split("_")[-1]
    return df
        
    

# sentences as index and annotators as columns
def split_to_hits2(df, outfolder):
    answercols = get_answer_cols(df)
    #answercols = [i[len(ANSWER_PREFIX):] for i in answercols]
    necessary_cols = ["workerid"] + answercols
    
    hits = list(set(df["hitid"].tolist()))  # get unique hits
    
    hitcount = 0
    for hit in hits:
        print "processing ",hitcount," ",hit," ",
        hitdf = df[df["hitid"] == hit]
        hitdf = hitdf[necessary_cols]
        hitdf = hitdf.dropna(axis=1, how="all")
        hitdf = hitdf.set_index("workerid")
        hitdf = hitdf.fillna(value="-")
        hitdf = strip_answers(hitdf)
        thitdf = hitdf.transpose()
        print " dim: ",hitdf.shape
        hitcsvpath = os.path.join(outfolder, hit + ".csv")
        IOtools.tocsv(thitdf, hitcsvpath, keepindex=True)
        

def get_file_sentence_map(df, remove_comments):
    filenames = get_filenames(df, remove_comments)
    answercols = get_answer_cols(df, remove_comments)
    filesentencemap = {} # gather the sentence_names under their respective filenames
    for fname in filenames:
        members = []
        for sentence_name in answercols:
            if sentence_name.startswith(fname):
                members.append(sentence_name)
        filesentencemap[fname] = members
    return filesentencemap



def get_mturkers_list(nmturkers=6):
    return ["MA"+str(i) for i in range(nmturkers)]


isanswer = lambda x : not x.endswith(nonsense_suffix)  # and not x.endswith(comment_suffix)


def get_answer_df(df, nmturkers=6):
    mturkers = get_mturkers_list(nmturkers)
    hdf = df[["docid", "sentenceid"] + mturkers]
    
    # remove comment and nonsense rows
    return hdf[hdf["sentenceid"].map(isanswer)]


def get_most_occ_in_list(votes):
    rank = Counter(votes) 
    winner = rank.most_common(1)[0]
    winnervote = winner[1]
    winnername = winner[0]
    if winnervote == 1:
        return EQUAL_VOTES
    else:
        return winnername

def find_majority_vote(indf, outfilepath):
    '''
    cat1 = ["MA0", "MA1", "MA2"]
    cat2 = ["MA3", "MA4", "MA5"]
    '''
    df = indf.copy()
    indices = df.index.tolist()
    j = 0
    for catname, mturkers in cats.iteritems():
        for i in indices:
            sentence = df.loc[i, "sentenceid"]
            print j, ": ", i,"   ",sentence,"  ",indf.shape
            if not sentence.endswith(comment_suffix):
                mturk_answers = df.loc[i, mturkers].tolist()  # [(hitid : answer)]
                mturk_answers = [answer.split(ANSWERCELL_SEP)[1] for answer in mturk_answers] # extract answer 
                majority_answer = get_most_occ_in_list(mturk_answers)
                finalanswercol = mvote_colname + catname
                df.loc[i, finalanswercol] = majority_answer
        j += 1
    IOtools.tocsv(df, outfilepath)
    

def insert_texts(df, sourcedf, outfilepath):
    sindices = sourcedf.index.tolist()
    c = 0
    for i in sindices:
        docid = sourcedf.loc[i, "docid"]
        sentenceid = sourcedf.loc[i, "sentenceid"]
        label = sourcedf.loc[i, "goldlabel"]
        text = sourcedf.loc[i, "text"]
        df.loc[(df["docid"] == docid) & (df["sentenceid"] == sentenceid), "text"] = text
        df.loc[(df["docid"] == docid) & (df["sentenceid"] == sentenceid), "GOLD"] = label
        if len(df.loc[(df["docid"] == docid) & (df["sentenceid"] == sentenceid), "GOLD"]) != 0:
            c += 1
    print c, " found"
    IOtools.tocsv(df, outfilepath)
    
    
  
def split_to_docs(df, outfolder):
    
    filesentencemap = get_file_sentence_map(df) # gather the sentence_names under their respective filenames
    '''
    filenames = get_filenames(df)
    answercols = get_answer_cols(df)
    
    for fname in filenames:
        members = []
        for sentence_name in answercols:
            if sentence_name.startswith(fname):
                members.append(sentence_name)
        filesentencemap[fname] = members
    '''
    
    for filename, sentences in filesentencemap.iteritems():
        
        filedf = df[["hitid", "assignmentid", "workerid"] + sentences] # annotations for the sentences of one file
        filedf = filedf.dropna(axis=0, how="all", subset=sentences) # clear unrelated rows, those contain no annotations for these sentences
        annotationcsvpath = os.path.join(outfolder, filename + ".csv")
        IOtools.tocsv(filedf, annotationcsvpath)
    

def reindex_df(sdf):
    oldindices = sdf.index.tolist()
    imap = {}
    for i,ind in enumerate(oldindices):
        imap[ind] = i
    return sdf.rename(index=imap)


def reindex_df_with(df, newindices):
    oldindices = df.index.tolist()
    imap = {}
    for o,n in zip(oldindices, newindices):
        imap[o] = n
    return df.rename(index=map) 


    
def get_annotation_matrix(df, outfolder, 
                          noworkerid=True,
                          includehitid=True,
                          nMturkAnnotators=7, nExpertAnnotators=2, 
                          nMturkFinalCol=1, nExperFinalCol=1,
                          remove_comments=False):
    filesentencemap = get_file_sentence_map(df, remove_comments)
    
    colnames = ["docid", "sentenceid"]
    mturkercols = ["MA"+str(i) for i in range(nMturkAnnotators)]
    mturk_final_col = ["Mturk_final"]
    expertcols = ["EA"+str(i) for i in range(nExpertAnnotators)]
    expert_final_col = ["GOLD"]
    
    colnames += mturkercols + mturk_final_col + expertcols + expert_final_col
    
    matrix = []
    for filename, sentences in filesentencemap.iteritems():
        filedf = df[["hitid", "workerid"] + sentences] # annotations for the sentences of one file
        filedf = filedf.dropna(axis=0, how="all", subset=sentences) # clear unrelated rows, those contain no annotations for these sentences
        filedf = reindex_df(filedf)
        filedf = filedf.fillna(value="NaN")
        
        for sentenceid in sentences:
            fileid = filename[len(ANSWER_PREFIX)-1:]
            trimmed_sentenceid = sentenceid[len(ANSWER_PREFIX)-1:]
            line = [fileid, trimmed_sentenceid]
            
            mturk_answers = []
            nannotators = filedf.shape[0]
            for i in range(nannotators):
                annotatorid = filedf.loc[i, "workerid"]
                hitid = filedf.loc[i, "hitid"]
                answer = filedf.loc[i, sentenceid]
                answer = answer.split("_")[-1]
                
                print "types: ", type(annotatorid), " ", type(answer), "  ", answer 
                answercell = annotatorid + ANSWERCELL_SEP + answer
                if noworkerid:
                    answercell = answer
                if includehitid and noworkerid:
                    if sentenceid.endswith(comment_suffix):
                        answercell = annotatorid + ANSWERCELL_SEP + answer 
                    #elif sentenceid.endswith(nonsense_suffix):
                        #answercell = answer
                    else:
                        answercell = hitid + ANSWERCELL_SEP + answer
                mturk_answers.append(answercell)
            line.extend(mturk_answers)
            line.extend(np.zeros(len(colnames)-len(line), dtype=int).tolist())  # fill 0's for the yet unknown cols (expert labels)
            matrix.append(line)
    
    annotationdf = pd.DataFrame(matrix, columns=colnames)
    outcsvpath = os.path.join(outfolder, "annotationdf_whitid_wcomments_noworkerid.csv")
    IOtools.tocsv(annotationdf, outcsvpath)
                

def get_annotator_names(df, nworkers=6):
    allids = []
    workerid_cols = ["MA"+str(i)+"_workerid" for i in range(nworkers)]
    indices = df.index.tolist()
    for i in indices:
        sentence = df.loc[i, "sentenceid"]
        if not sentence.endswith(comment_suffix):
            workeranswerpairs = df.loc[i, workerid_cols].tolist()
            #print workeranswerpairs
            workerids = [j.split(ANSWERCELL_SEP)[0] for j in workeranswerpairs]
            allids.extend(workerids) 
    return list(set(allids))


def initialize_map(keys):
    m = {}
    for k in keys:
        m[k] = 0
    return m

def find_annotator_disagreements(df, outpath):
    annotators = get_annotator_names(df)
    
    majority_disagr = initialize_map(annotators)
    gold_disagr = initialize_map(annotators)
    numofannotations = initialize_map(annotators)
    
    indices = df.index.tolist()
    for i in indices:
        gold_ans = df.loc[i, "GOLD"]
        sentence = df.loc[i, "sentenceid"]
        if not sentence.endswith(comment_suffix):
            for j1 in range(0, 3):   # compare cat1 answers
                major_ans = df.loc[i, mvote_colname+"cat1"]
                workeranswerpair = df.loc[i, "MA"+str(j1)+"_workerid"]
                print workeranswerpair
                items = workeranswerpair.split(ANSWERCELL_SEP)
                worker = items[0].strip()
                answer = items[1].strip()
                
                print answer,"  ",major_ans,"  ",gold_ans
                if answer != major_ans:
                    majority_disagr[worker] = majority_disagr[worker] + 1
                if answer != gold_ans:
                    gold_disagr[worker] = gold_disagr[worker] + 1
                numofannotations[worker] = numofannotations[worker] + 1
            
            for j1 in range(3, 6):   # compare cat1 answers
                major_ans = df.loc[i, mvote_colname+"cat2"]
                workeranswerpair = df.loc[i, "MA"+str(j1)+"_workerid"]
                items = workeranswerpair.split(ANSWERCELL_SEP)
                worker = items[0].strip()
                answer = items[1].strip()
                
                print answer,"  ",major_ans,"  ",gold_ans
                if answer != major_ans:
                    majority_disagr[worker] = majority_disagr[worker] + 1
                if answer != gold_ans:
                    gold_disagr[worker] = gold_disagr[worker] + 1
                numofannotations[worker] = numofannotations[worker] + 1
    
    cols = ["annotatorid", "nMajorityDisagr", "nGoldDisagr", "nAnnotations", "weightedMajorityDisagr", "weightedGoldDisagr"]
    matrix = np.zeros([len(annotators), len(cols)], dtype=object)
    adf = pd.DataFrame(matrix, index=range(len(annotators)), columns=cols)
    for i,worker in enumerate(annotators):
        adf.loc[i, "annotatorid"] = worker
        adf.loc[i, "nMajorityDisagr"] = majority_disagr[worker]
        adf.loc[i, "nGoldDisagr"] = gold_disagr[worker]
        adf.loc[i, "nAnnotations"] = numofannotations[worker]
        if numofannotations[worker] == 0:
            adf.loc[i, "weightedMajorityDisagr"] = -1
            adf.loc[i, "weightedGoldDisagr"] = -1
        else:
            adf.loc[i, "weightedMajorityDisagr"] = round(majority_disagr[worker] / float(numofannotations[worker]), 4)
            adf.loc[i, "weightedGoldDisagr"] = round(gold_disagr[worker] / float(numofannotations[worker]), 4)
    
    IOtools.tocsv(adf, outpath)
     

def run_annotationmatrix():
    csvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/fallacy-task_mturk_gold-annotated.csv"
    df = IOtools.readcsv(csvpath, False)
    annotationoutfolder = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/"
    get_annotation_matrix(df, annotationoutfolder)  

def run_splits():
    csvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/fallacy-task_mturk_gold-annotated.csv"
    df = IOtools.readcsv(csvpath, False)
    hitsoutfolder = "/home/dicle/Dropbox/ukp/fallacy_detection/hitcsvs"
    split_to_hits(df, hitsoutfolder)

    docsoutfolder = "/home/dicle/Dropbox/ukp/fallacy_detection/doccsvs/"
    split_to_docs(df, docsoutfolder)


def run_majority_vote():
    incsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_whitid_wcomments_noworkerid.csv"
    indf = IOtools.readcsv(incsvpath)
    outcsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_whitid_wmajority_wcomm.csv"
    find_majority_vote(indf, outcsvpath)    

def run_copy_from_gold():
    maincsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_worker.csv"
    indf = IOtools.readcsv(maincsvpath)
    sourcecsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/expertandgoldannotations/gold-labels3.csv"
    sourcedf = IOtools.readcsv(sourcecsvpath)
    outfilepath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_wtexts_wmajority_worker.csv"
    insert_texts(indf, sourcedf, outfilepath)    
    


if __name__ == '__main__':
    
    csvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/fallacy-task_mturk_gold-annotated.csv"
    df = IOtools.readcsv(csvpath, False)
    hitsoutfolder = "/home/dicle/Dropbox/ukp/fallacy_detection/hitcsvs_transposed2/"
    split_to_hits2(df, hitsoutfolder)
    
    '''
    maincsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_wtexts_wmajority_worker.csv"
    indf = IOtools.readcsv(maincsvpath)
    annotatordfpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/mturker_agr_rates.csv"
    find_annotator_disagreements(indf, annotatordfpath)
    '''
    
    
    '''
    maincsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_worker.csv"
    indf = IOtools.readcsv(maincsvpath)
    sourcecsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/expertandgoldannotations/gold-labels3.csv"
    sourcedf = IOtools.readcsv(sourcecsvpath)
    outfilepath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_wtexts_wmajority_worker.csv"
    insert_texts(indf, sourcedf, outfilepath)
    '''
    '''
    incsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_whitid_wcomments_noworkerid.csv"
    indf = IOtools.readcsv(incsvpath)
    outcsvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/annotationdf_whitid_wmajority_wcomm.csv"
    find_majority_vote(indf, outcsvpath)
    '''
    
    '''
    csvpath = "/home/dicle/Dropbox/ukp/fallacy_detection/fallacy-task_mturk_gold-annotated.csv"
    df = IOtools.readcsv(csvpath, False)
    
    print df.shape
    print "ncols: ", df.iloc[0, :].size
    
    answercols = get_answer_cols(df)
    print answercols[0],"  ",df.loc[0, "Answer.sT6_sID565_cID8995_1"]
    print 'h'
    
     
    hitsoutfolder = "/home/dicle/Dropbox/ukp/fallacy_detection/hitcsvs"
    split_to_hits(df, hitsoutfolder)

    docsoutfolder = "/home/dicle/Dropbox/ukp/fallacy_detection/doccsvs/"
    split_to_docs(df, docsoutfolder)
    
    
    annotationoutfolder = "/home/dicle/Dropbox/ukp/fallacy_detection/mturk_annotations/"
    get_annotation_matrix(df, annotationoutfolder)
    '''
    
    pass