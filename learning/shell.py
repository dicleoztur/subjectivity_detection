'''
Created on Sep 3, 2014

@author: dicle
'''

import os

from sentimentfinding import IOtools
import learner_selected
import performance_evaluation_crossval




def cross_featuresets():
    
    path = "/home/dicle/Dicle/Tez/corpusstats/learning11/cross_features/nb2"
    
    featset = {'redef-rat_lex-tf': 'comb4030_F_0-3_1-1_2-1_3-2_4-3_5-1_6-1_7-0_8-2', 
               'redef-rat_lex-pres': 'comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1',
               'redef-rat_lex-rat': 'comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3',
               'redef-rat_lex-cnt': 'comb1820_F_0-1_1-1_2-1_3-0_4-1_5-1_6-1_7-0_8-0'}
    '''labelunions = ['ALLobj-STGsubj_NC-2',
                   'STGobj-ALLsubj_NC-2',
                   'WKobj-WKsubj_NC-2',
                   'EACHobj-EACHsubj_NC-4']'''
    labelunions = ['STGobj-STGsubj_NC-2',
                   'ALLobj-STGsubj_NC-2',
                   'ALLobj-ALLsubj_NC-2',
                   'STGobj-ALLsubj_NC-2',                 
                   'WKobj-WKsubj_NC-2',
                   'EACHobj-EACHsubj_NC-4']
    
    for lunion in labelunions:
        for featclass,combname in featset.iteritems():
            learner_selected.experiment(rootpath=path, featureclass=featclass, combfilename=combname, labelunion=lunion)
            performance_evaluation_crossval.evaluate_crossfeatures(scoresroot=path)
            
            

def cross_corpus():
    
    #path = "/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_equaltest/scores"
    path = "/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_test_collision/EQUALsize_full"
    for i in range(1,6):
        p = IOtools.ensure_dir(path + "/" + str(i))
         
        learner_selected.split_for_cross_corpus(rootpath=p, equalsize=True)
        performance_evaluation_crossval.evaluate_crosscorpus(scoresroot=p)


def char_ngram_classification(charngram_csv_fname, charngramXpath, featureclass='redef-rat_lex-rat'):
    
    scoresrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/charngram/"

    
    labelunions = ['STGobj-STGsubj_NC-2',
                   'ALLobj-STGsubj_NC-2',
                   'ALLobj-ALLsubj_NC-2',
                   'STGobj-ALLsubj_NC-2',                 
                   'WKobj-WKsubj_NC-2',
                   'EACHobj-EACHsubj_NC-4']
    
    #labelunions = ['EACHobj-EACHsubj_NC-4']
    scorespath = IOtools.ensure_dir(os.path.join(scoresrootpath, charngram_csv_fname[:-4]))
    
    for lunion in labelunions:
        print scorespath
        
        
        learner_selected.charngram_classification(rootpath=scorespath, 
                                                  charngram_csv_fname=charngram_csv_fname,
                                                  charngramXpath=charngramXpath,
                                                  featureclass=featureclass,
                                                  labelunion=lunion)
        
        foldrootpath = os.path.join(scorespath, featureclass, lunion[:-5])
        # performance evaluation
        performance_evaluation_crossval.evaluate_charngrams(foldrootpath, annottype="double", 
                                                            featset=featureclass, labelunion=lunion)



def experiment_features(dfpath, featureclass, scorespath, nfolds=5):
    labelunions = ['STGobj-STGsubj_NC-2',
                   'ALLobj-STGsubj_NC-2',
                   'ALLobj-ALLsubj_NC-2',
                   'STGobj-ALLsubj_NC-2',                 
                   'WKobj-WKsubj_NC-2',
                   'EACHobj-EACHsubj_NC-4']
    
      
    for lunion in labelunions:
        print scorespath
        
        
        learner_selected.standard_classification(rootpath=scorespath, 
                                                 Xpath=dfpath, 
                                                 featureclass=featureclass, 
                                                 labelunion=lunion, 
                                                 nfolds=nfolds)
        
        foldrootpath = os.path.join(scorespath, featureclass, lunion[:-5])
        # performance evaluation
        performance_evaluation_crossval.evaluate_charngrams(foldrootpath, annottype="double", 
                                                            featset=featureclass, labelunion=lunion)
        
        

def term_binary_presence_classification(binpres_df_path, featureclass='binpres', nfolds=10):
    
    scorespath = "/home/dicle/Dicle/Tez/corpusstats/learning11/binpresence/scores_rbfkernel/"

    
    labelunions = ['STGobj-STGsubj_NC-2',
                   'ALLobj-STGsubj_NC-2',
                   'ALLobj-ALLsubj_NC-2',
                   'STGobj-ALLsubj_NC-2',                 
                   'WKobj-WKsubj_NC-2',
                   'EACHobj-EACHsubj_NC-4']
    
      
    for lunion in labelunions:
        print scorespath
        
        
        '''learner_selected.binarypresence_classification(rootpath=scorespath, 
                                                       binpresXpath=binpres_df_path, 
                                                       featureclass=featureclass, 
                                                       labelunion=lunion, 
                                                       nfolds=nfolds)
        '''
        foldrootpath = os.path.join(scorespath, featureclass, lunion[:-5])
        # performance evaluation
        performance_evaluation_crossval.evaluate_charngrams(foldrootpath, annottype="double", 
                                                            featset=featureclass, labelunion=lunion)
        

def bag_of_words_baseline_classification(featureclass='redef-rat_lex-rat',
                                         bow_csv_fname='contenttermTFIDF.csv'):
    
    scoresrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/BoWbaseline"

    
    labelunions = ['STGobj-STGsubj_NC-2',
                   'ALLobj-STGsubj_NC-2',
                   'ALLobj-ALLsubj_NC-2',
                   'STGobj-ALLsubj_NC-2',                 
                   'WKobj-WKsubj_NC-2',
                   'EACHobj-EACHsubj_NC-4']
    
    #labelunions = ['EACHobj-EACHsubj_NC-4']
    scorespath = IOtools.ensure_dir(os.path.join(scoresrootpath, bow_csv_fname[:-4]))
    
    for lunion in labelunions:
        
        learner_selected.bog_baseline_classification(rootpath=scorespath, 
                                                     bow_csv_fname=bow_csv_fname,
                                                     featureclass=featureclass,
                                                     labelunion=lunion)
    
        foldrootpath = os.path.join(scorespath, featureclass, lunion[:-5])
        #foldrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/bogbaseline/redef-rat_lex-rat/ALLobj-ALLsubj/bogfeatures_baseline"
        performance_evaluation_crossval.evaluate_bogbaseline(foldpath=foldrootpath)
    
    
    
def run_cross_validation(outrootpath):
    k = 5
    #scorespath = os.path.join(outrootpath, "scores")
    #learner_selected.conduct_cross_validation_notest(k, outrootpath)
    performance_evaluation_crossval.evaluate_cross_validation(outrootpath)
    



def crawl_folds_for_sets(foldpath, outrootpath):
    folds = IOtools.getfoldernames_of_dir(foldpath)  
    trainitems_fname = "trainitems.csv"
    testitems_fname = "testitems.csv"
    items = {"trainitems.csv" : [], "testitems.csv" : []} 
    for fold in folds:
        p1 = os.path.join(foldpath, fold)
        for fname in items.keys():
            p2 = os.path.join(p1, fname)
            df = IOtools.readcsv(p2, keepindex=True)
            fileids = df.index.values.tolist()
            outpath = os.path.join(outrootpath, "all-"+fname[:-4]+".txt")
            IOtools.todisc_list(outpath, fileids, mode='a')
            items[fname].extend(fileids)
    return items
        
        

def test_sets():
    
    # collect crosscorpus train1-test1
    initpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/5fold_majoritybaseline/scores/double/redef-rat_lex-rat/comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3/ALLobj-ALLsubj"
    initoutpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/corpuscheck/inititems/"
    crawl_folds_for_sets(initpath, initoutpath)
    
    crosscorpinpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_equaltest/scores_equalsize2/redef-rat_lex-rat/ALLobj-ALLsubj/test_with_corpus1/train_corpus1"
    crosscorpoutpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/corpuscheck/crosscorpusitems"
    crawl_folds_for_sets(crosscorpinpath, crosscorpoutpath)
    
    
    
    initpath = ""


def run_charngram_experiment():
    
    csvfolder = "/home/dicle/Dicle/Tez/stylistic_comparison/test1/output/"
    csvs = ["superchar-2gram.csv"]
    
    for charngramcsvfname in csvs:
        p = os.path.join(csvfolder, charngramcsvfname)
        char_ngram_classification(charngram_csv_fname=charngramcsvfname, charngramXpath=p, featureclass='redef-rat_lex-rat')



def run_term_binary_presence_experiment():
    binpres_df_path = "/home/dicle/Dicle/Tez/corpusstats/learning11/binpresence/contenttermPRES.csv"
    term_binary_presence_classification(binpres_df_path, featureclass="binpres", nfolds=10)
    
    
# for comparison with Scholz, Conrad&Hillekamps2012
def run_tfidf_features_experiment():
    dfpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/tfidf/contenttermTFIDF.csv"
    featureclass = "tfidf"
    scorespath = "/home/dicle/Dicle/Tez/corpusstats/learning11/tfidf/scores/"
    experiment_features(dfpath, featureclass, scorespath, nfolds=5)
    
    

def run_bow_baseline():
    dfpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/BoWbaseline/contenttermCOUNT/contenttermCOUNT.csv"
    scorespath = "/home/dicle/Dicle/Tez/corpusstats/learning11/BoWbaseline/contenttermCOUNT/scores/"
    featureclass = "bow"
    experiment_features(dfpath, featureclass, scorespath, nfolds=5)

if __name__ == "__main__":
    
    '''
    outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/5fold_majoritybaseline"
    run_cross_validation(outrootpath)
    '''
    
    #cross_corpus()
    #bag_of_words_baseline_classification()
    
    #test_sets()
    
    #cross_featuresets()

    #run_charngram_experiment()
    #run_term_binary_presence_experiment()
    #run_tfidf_features_experiment()
    run_bow_baseline()
    
    
    