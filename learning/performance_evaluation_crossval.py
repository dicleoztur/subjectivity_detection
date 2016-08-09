'''
Created on Jul 3, 2014

@author: dicle
'''
'''
Created on Apr 14, 2014

@author: dicle
'''


import os
import pandas as pd
import numpy as np

import metaexperimentation, matrixhelpers, utils
from sentimentfinding import IOtools


class PerformanceEvaluator:
    experimentspath = ""
    
    def __init__(self, expspath=metaexperimentation.experimentsrootpath, takeworst=False):
        self.experimentspath = expspath
        self.resultspath = IOtools.ensure_dir(os.path.join(self.experimentspath, "performance"))
        self.N = 50   # ranking number
        #self.metricname = basemetric
        self.prefix = "best"
        self.takeworst = takeworst
        if takeworst:
            self.prefix = "worst"
            
    
    def best_score_per_fold(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
                        
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                   
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    
                    for labelunion in labelunions:
                    
                        p4 = os.path.join(p3, labelunion)  
                        folds = IOtools.getfoldernames_of_dir(p4)
                        
                        for fold in folds:
                            
                            p5 = os.path.join(p4, fold)                       
                            print p5
                            annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                               
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            # drop clustering results as they are useless being not worked on (back validation missing)
                            scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",annotdf.shape
                            rankdf["labelunion"] = labelunion
                            rankdf["featureset"] = metricclass + " ** " + combname
                            rankdf["annottype"] = annottype
                            rankdf["fold"] = fold
                            #dflist.append(rankdf)
                            annotdf = annotdf.append(rankdf)
                            print scorecsvfile.shape
                                                           
                                #bigdf = bigdf.append(rankdf)  # bigdf header = metaexperimentation.performanceheader
                            
                            
                            #annotdf["nclasses"] = nclass
                        
                        #annotdf["featureset"] = combname
                    
                    #annotdf["size"] = setsize
                
                #annotdf["annottype"] = annottype  
                            print " * ",annotdf.shape
                
                            annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname], ascend=self.takeworst)  
                
                            bigdf = bigdf.append(annotdf)
            # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["fold", metricname], ascending=self.takeworst, inplace=True)
        bigdf = bigdf.groupby("fold").head(self.N)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_fold-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
        
    
    def best_score_per_annottype(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                   
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    
                    for labelunion in labelunions:
                    
                        p4 = os.path.join(p3, labelunion)  
                        folds = IOtools.getfoldernames_of_dir(p4)
                        
                        for fold in folds:
                            
                            p5 = os.path.join(p4, fold)                       
                                
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            # drop clustering results as they are useless being not worked on (back validation missing)
                            scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",annotdf.shape
                            rankdf["labelunion"] = labelunion
                            rankdf["featureset"] = metricclass + " ** " + combname
                            rankdf["annottype"] = annottype
                            #dflist.append(rankdf)
                            annotdf = annotdf.append(rankdf)
                            print scorecsvfile.shape
                
                            annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname], ascend=self.takeworst)  
                
                            bigdf = bigdf.append(annotdf)
            # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["annottype", metricname], ascending=self.takeworst, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_annottype-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
    # best alg is the same. just output alg name.                                                        
                
    
    
    
    def best_score_per_labelunion(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
                       
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                    
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)                
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                                
                    for labelunion in labelunions:
                    
                        luniondf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                    
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        # drop clustering results as they are useless being not worked on (back validation missing)
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                        
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                        print rankdf.shape
                        #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                        print " ** ",luniondf.shape
                        rankdf["labelunion"] = labelunion
                        rankdf["featureset"] = metricclass + " ** " + combname
                        rankdf["annottype"] = annottype
                        #dflist.append(rankdf)
                        luniondf = luniondf.append(rankdf)
                        print scorecsvfile.shape
                                                           
                        luniondf = matrixhelpers.get_first_N_rows(luniondf, self.N, [metricname], ascend=self.takeworst)  
                        bigdf = bigdf.append(luniondf)
                    
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["labelunion", metricname], ascending=self.takeworst, inplace=True)
        
        # 14Nisan added new to get top N
        bigdf = bigdf.groupby("labelunion").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_labelunion-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
 
 
    
    def best_score_per_featureset(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
                       
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                     
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)                
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    combdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                             
                    for labelunion in labelunions:
                    
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        # drop clustering results as they are useless being not worked on (back validation missing)                        
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                                                                                 
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                        print rankdf.shape
                        #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                        print " ** ",combdf.shape
                        rankdf["labelunion"] = labelunion
                        rankdf["featureset"] = metricclass + " ** " + combname
                        rankdf["annottype"] = annottype
                        #dflist.append(rankdf)
                        combdf = combdf.append(rankdf)
                        print scorecsvfile.shape
                                                           
                    combdf = matrixhelpers.get_first_N_rows(combdf, self.N, [metricname], ascend=self.takeworst)  
                    bigdf = bigdf.append(combdf)
                
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["featureset", metricname], ascending=self.takeworst, inplace=True)
        
        # 14Nisan added new to get top N
        bigdf = bigdf.groupby("featureset").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_featureset-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
        
           
        
    
    def best_score_per_algorithm(self, metricname, scorepath=metaexperimentation.expscorepath):
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            #annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                           
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    
                    for labelunion in labelunions:
                        
                        annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                        
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        # drop clustering results as they are useless being not worked on (back validation missing)
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                        
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                        print rankdf.shape
                        #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                        print " ** ",annotdf.shape
                        rankdf["labelunion"] = labelunion
                        rankdf["featureset"] = metricclass + " ** " + combname
                        rankdf["annottype"] = annottype
                        #dflist.append(rankdf)
                        annotdf = annotdf.append(rankdf)
                        print scorecsvfile.shape
                
                        annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname], ascend=self.takeworst)  
                        bigdf = bigdf.append(annotdf)
            
        bigdf.sort(["algorithm", metricname], ascending=self.takeworst, inplace=True)
        bigdf = bigdf.groupby("algorithm").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_algorithm-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
  
  

  
    def score_stats(self, metricname, scorepath=metaexperimentation.expscorepath):
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
                
        for annottype in annottypes:
                        
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                           
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    
                    for labelunion in labelunions:
                        
                        p4 = os.path.join(p3, labelunion)          
                        folds = IOtools.getfoldernames_of_dir(p4)
                        
                        for fold in folds:
                            
                            p5 = os.path.join(p4, fold)
                        
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath) 
                            
                            # drop clustering results as they are useless being not worked on (back validation missing)
                            scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                            
                            scorecsvfile["labelunion"] = labelunion
                            scorecsvfile["featureset"] = metricclass + " ** " + combname
                            scorecsvfile["annottype"] = annottype
                            scorecsvfile["fold"] = fold
                            bigdf = bigdf.append(scorecsvfile)
        
        
        print "bigdf ",bigdf.shape
        print bigdf.columns
        print
        
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, "bigdf.csv"))
        
        groups = ["annottype", "labelunion", "algorithm", "featureset", "fold"]
        for factor in groups:
            
            grouped = bigdf.groupby(factor)
            dfgavg = grouped.mean()
            dfgmin = grouped.min()
            dfgmax = grouped.max()
            dfgstd = grouped.std()
            
            
            nrows, _ = dfgavg.shape
            outdf = pd.DataFrame(np.zeros((nrows, 4)), index=dfgavg.index.values.tolist(), columns=["mean", "min", "max", "std"])
            
            outdf["mean"] = dfgavg[metricname]
            outdf["min"] = dfgmin[metricname]
            outdf["max"] = dfgmax[metricname]
            outdf["std"] = dfgstd[metricname]
            
            filename = "stats_"+factor+"-"+metricname.upper()
            IOtools.tocsv(outdf, os.path.join(self.resultspath, filename+".csv"), keepindex=True)
            


 
    
    '''
    def best_algorithm_per_annottype(self):
        
        return
    
    def best_algorithm_per_size(self):
        
        return
    
    def best_algorithm_per_nclasses(self):
        return
    
    def best_algorithm_per_combination(self):
        return   
    
    def best_algorithm_per_tagger(self):
        return
    '''
    
    def mean_scores(self):
        return


def evaluate_featureexcluded_datasets():
    rootpath = "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/experiments/"
    metrics = ["accuracy", "fscore", "precision", "recall"]
    
    scorespath = os.path.join(rootpath, "scores")
    exclusiontypes = IOtools.getfoldernames_of_dir(scorespath)
    
    '''
    for exclusionname in exclusiontypes:
        inputscorespath = os.path.join(scorespath, exclusionname)
        recordpath = os.path.join(rootpath, exclusionname)
        
        for minmax in [True, False]: 
            evaluator = PerformanceEvaluator(expspath=recordpath, takeworst=minmax)            
            for metric in metrics:
                print
                
                evaluator.best_score_per_algorithm(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_annottype(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_featureset(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_labelunion(metricname=metric, scorepath=inputscorespath)
    '''

    
    for exclusionname in exclusiontypes:
        rankpath = os.path.join(rootpath, exclusionname)
        inputscorespath = os.path.join(scorespath, exclusionname)
        
        evaluator = PerformanceEvaluator(expspath=rankpath, takeworst=True)
        for metric in metrics:
            evaluator.score_stats(metricname=metric, scorepath=inputscorespath)
            
            


def evaluate_fullsets():
    rootpath = "/home/dicle/Dicle/Tez/corpusstats/"
    folders = ["learning9", "learning9_svmscale", "learning9_noscale"]
    metrics = ["accuracy", "fscore", "precision", "recall"]
    
    for foldername in folders:
        
        recordpath = os.path.join(rootpath, foldername, "experiments")
        inputscorespath = os.path.join(rootpath, foldername, "experiments", "scores")
        
        for minmax in [True, False]:
            
            evaluator = PerformanceEvaluator(expspath=recordpath, takeworst=minmax)            
            for metric in metrics:
                print
                # done
                evaluator.best_score_per_algorithm(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_annottype(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_featureset(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_labelunion(metricname=metric, scorepath=inputscorespath)

    
    for foldername in folders:
        
        recordpath = os.path.join(rootpath, foldername, "experiments")
        inputscorespath = os.path.join(rootpath, foldername, "experiments", "scores")
        
        evaluator = PerformanceEvaluator(expspath=recordpath, takeworst=minmax)            
        for metric in metrics:
            print metric,"  0000000"
            evaluator.score_stats(metricname=metric, scorepath=inputscorespath)
            


def evaluate_cross_validation(rootpath):
    
    metrics = ["accuracy", "fscore", "precision", "recall"]
    
    #rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/experiments_10fold_scale/"
    #rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/exp-weaklabels/crossval/"
    
    inpath = os.path.join(rootpath, "scores")
    outpath = os.path.join(rootpath, "performance") 
    
    for minmax in [True, False]:
        evaluator = PerformanceEvaluator(expspath=rootpath, takeworst=minmax)
        for metric in metrics:
            evaluator.best_score_per_fold(metric)
    
    evaluator = PerformanceEvaluator(expspath=rootpath)
    for metric in metrics:
        evaluator.score_stats(metric)
        
    get_fold_averages(outpath)



def get_fold_averages_ablation():
    ablationCVscoresroot = "/home/dicle/Dicle/Tez/corpusstats/learning11/ablation2/"
    ablationtypes = ["item", "group", "onedim"]
    
    annotationtypes = ["double"]
    featsets = ["redef-rat_lex-rat"]
    '''labelunions = ["EACHobj-EACHsubj","ALLobj-ALLsubj","ALLobj-STGsubj", 
               "STGobj-ALLsubj", "STGobj-STGsubj", "WKobj-WKsubj"]
    '''
    
    
    for ablationtype in ablationtypes:
        
        print ablationtype
        
        p1 = os.path.join(ablationCVscoresroot, ablationtype, "scores")
        
        exclusionnames = IOtools.getfoldernames_of_dir(p1)
        
        for excname in exclusionnames:
            
            bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p2 = os.path.join(p1, excname)
            
            for annottype in annotationtypes:
                p3 = os.path.join(p2, annottype)
                
                for featset in featsets:
                    p4 = os.path.join(p3, featset)
                    combname = IOtools.getfoldernames_of_dir(p4)[0] # we know that there is only one folder
                    
                    p5 = os.path.join(p4, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p5)
                    
                    for labelunion in labelunions: 
                        p6 = os.path.join(p5, labelunion)
                        
                        folds = IOtools.getfoldernames_of_dir(p6)
                        
                        for foldno in folds:
                            p7 = os.path.join(p6, foldno)
                                                        
                            scorecsvfilepath = p7 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            
                            print " scorefile ",scorecsvfilepath,"  ",scorecsvfile.shape
                            
                            #rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(N / 2), metricnames, ascend=takeworst)
                            rankdf = scorecsvfile.copy()
                            rankdf["labelunion"] = labelunion
                            rankdf["featureset"] = featset + " ** " + combname
                            rankdf["annottype"] = annottype
                            rankdf["fold"] = foldno
                            #dflist.append(rankdf)
                            bigdf = bigdf.append(rankdf)
    
            print bigdf.shape,"  ",p2
            IOtools.tocsv(bigdf, os.path.join(p2, "bigdf.csv"))
            get_fold_averages(p2)
    
            
    
def print_accuracy_ablation():
    ablationCVscoresroot = "/home/dicle/Dicle/Tez/corpusstats/learning11/ablation2/"
    ablationtypes = ["item", "group", "onedim"]
    
    annotationtypes = ["double"]
    featsets = ["redef-rat_lex-rat"]
    '''labelunions = ["EACHobj-EACHsubj","ALLobj-ALLsubj","ALLobj-STGsubj", 
               "STGobj-ALLsubj", "STGobj-STGsubj", "WKobj-WKsubj"]
    '''
    
    inscorescsv = "fold_stats-ACCURACY.csv"
    
    for ablationtype in ablationtypes:
        
        print ablationtype
        
        p1 = os.path.join(ablationCVscoresroot, ablationtype, "scores")
        
        exclusionnames = IOtools.getfoldernames_of_dir(p1)
        
        for excname in exclusionnames:
            
            print excname           
            p2 = os.path.join(p1, excname)
            
            accdf = IOtools.readcsv(os.path.join(p2, inscorescsv), False)
            
            #filter for relevant lunions
            featset = "redef-rat_lex-rat ** comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"
            annottype = "(double"
            alg = "_MT-classification_alg-SVC_k-rbf_C-1)"
            lunions = ["EACHobj-EACHsubj","ALLobj-ALLsubj","ALLobj-STGsubj", 
                       "STGobj-ALLsubj", "STGobj-STGsubj", "WKobj-WKsubj"]
            # get mean accuracy and std
            #accdf["meanROUND"] = accdf.iloc[:, 4].values
            
            nrows, ncols = accdf.shape
            for l in lunions:
                rowname = ", ".join([annottype, featset,l, alg])
                rowname = rowname.strip().decode("utf8")
                print "q",rowname,"q 00 ",accdf.iloc[nrows-2,0]
                print type(rowname)," 00 ",type(accdf.iloc[5,0])
                print len(rowname)," 00 ",len(accdf.iloc[5,0])
                print rowname == accdf.iloc[nrows-2,0]
                xdf = accdf[accdf.iloc[:,0] == rowname]
                print l
                print "\t",xdf.loc[:, "accround"],"\t",xdf.loc[:, "stdround"]
                print
            print "\n\n"
            

def get_allfolds_bigdf(foldrootpath, annottype, featset, labelunion):
    
    bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
    
    folds = IOtools.getfoldernames_of_dir(foldrootpath)
                        
    for foldno in folds:
        p1 = os.path.join(foldrootpath, foldno)
                                    
        scorecsvfilepath = p1 + os.sep + metaexperimentation.scorefilename+".csv"
        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
        
        print " scorefile ",scorecsvfilepath,"  ",scorecsvfile.shape
        
        #rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(N / 2), metricnames, ascend=takeworst)
        rankdf = scorecsvfile.copy()
        rankdf["labelunion"] = labelunion
        rankdf["featureset"] = featset 
        rankdf["annottype"] = annottype
        rankdf["fold"] = foldno
        bigdf = bigdf.append(rankdf)
        #dflist.append(rankdf)
    
    
    print "FOLDROOTPATH ",foldrootpath
    outcsvpath = os.path.join(foldrootpath, "bigdf.csv")
    IOtools.tocsv(bigdf, outcsvpath, False)
    
        
        

def evaluate_crossfeatures(scoresroot):
    
    featclasses = IOtools.getfoldernames_of_dir(scoresroot)
    
    for featureclass in featclasses:
        
        p1 = os.path.join(scoresroot, featureclass)
        lunions = IOtools.getfoldernames_of_dir(p1)
        
        for labelunion in lunions:
            
            p2 = os.path.join(p1, labelunion)  # foldspath
      
            get_allfolds_bigdf(foldrootpath=p2, 
                               annottype=featureclass, 
                               featset=featureclass, 
                               labelunion=labelunion)
            
            get_fold_averages(p2)

       

def evaluate_crosscorpus(scoresroot):
    
    featclasses = IOtools.getfoldernames_of_dir(scoresroot)
    
    for featureclass in featclasses:
        
        p1 = os.path.join(scoresroot, featureclass)
        lunions = IOtools.getfoldernames_of_dir(p1)
        
        for labelunion in lunions:
            
            p2 = os.path.join(p1, labelunion)

            testcases = IOtools.getfoldernames_of_dir(p2)
            
            for testcase in testcases:
                
                p3 = os.path.join(p2, testcase)
                traincases = IOtools.getfoldernames_of_dir(p3)
                
                for traincase in traincases:
                    
                    p4 = os.path.join(p3, traincase)   # foldspath
                    get_allfolds_bigdf(foldrootpath=p4, 
                                       annottype=testcase + " ** "+traincase, 
                                       featset=featureclass, 
                                       labelunion=labelunion)
                    
                    get_fold_averages(p4)
                


def evaluate_bogbaseline(foldpath):
    get_allfolds_bigdf(foldrootpath=foldpath, 
                        annottype='double', 
                        featset='redef-rat_lex-rat', 
                        labelunion='ALLobj-ALLsubj_NC-2')
                    
    get_fold_averages(foldpath)
    
            
def evaluate_charngrams(foldpath, annottype, featset, labelunion):
    get_allfolds_bigdf(foldpath, annottype, featset, labelunion)         
    get_fold_averages(foldpath)        
            
            
def get_fold_averages(resultspath):
#def get_fold_averages(crossvalallresultspath):
    #crossvalallresultspath = "/home/dicle/Dicle/Tez/corpusstats/learning10/exp-weaklabels/crossval/performance/"
    
    bigdf = IOtools.readcsv(resultspath+os.sep+"bigdf.csv", False)
    
    print bigdf.shape," ",bigdf.columns.values
    print bigdf.iloc[0,[2,3]],"  ",bigdf.iloc[4,5]
    
    g1 = bigdf.groupby(["annottype", "featureset", "labelunion", "algorithm"])

    metrics = ["accuracy", "fscore", "precision", "recall"]
    
    
    dfg = g1.mean()
    
    
    #print dfg.columns,"  ",dfg.shape
    
    x = dfg.index.values.tolist()
   
    nrows, _ = dfg.shape
    
    for metric in metrics:
        outdf = pd.DataFrame(np.zeros((nrows, 4)), index=x, columns=["mean", "min", "max", "std"])
        
        meandf = g1.mean()
        mindf = g1.min()
        maxdf = g1.max()
        stddf = g1.std()
        
        outdf["mean"] = meandf[metric]
        outdf["min"] = mindf[metric]
        outdf["max"] = maxdf[metric]
        outdf["std"] = stddf[metric]
        
        filename = resultspath+os.sep+"fold_stats-"+metric.upper()
        print filename
        outdf["accround"] = np.around(outdf.loc[:,"mean"].values*100, 2)
        outdf["stdround"] = np.around(outdf.loc[:,"std"].values*100, 2)
        IOtools.tocsv(outdf, filename+".csv", True)            



if __name__ == "__main__":
    
    #evaluate_cross_validation()
    
    #ablation
    #get_fold_averages_ablation()
    #print_accuracy_ablation()
    
    #evaluate_fullsets()
    #evaluate_featureexcluded_datasets()            
                           
        
    #evaluator.best_score_per_annottype(metricname="accuracy")
    #evaluator.best_score_per_labelunion(metricname="fscore")
    #evaluator.best_score_per_featureset(metricname="precision")
    #evaluator.best_score_per_algorithm(metricname="accuracy")
    
    '''
    evaluator.best_score_per_size(metricname="precision")
    evaluator.best_score_per_nclasses(metricname="precision")
    evaluator.best_score_per_dataset(metricname="accuracy")
    evaluator.best_score_per_tagger(metricname="fscore")
    '''
    
    #evaluate_cross_validation(rootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/5fold_test20p/")
    evaluate_crosscorpus(scoresroot="/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_equaltest/scores_equalsize/")

