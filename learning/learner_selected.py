'''
Created on Jun 27, 2014

@author: dicle
'''


from sklearn import cluster
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing, metrics

import math
import random
import numpy as np
import os
import time, multiprocessing


import metaexperimentation, utils
from corpus import metacorpus
from sentimentfinding import IOtools
from txtprocessor import listutils



def handler(signum, frame):
    print "    not forever"
    raise Exception("quitting the training")



class LearningExperiment:
    
    experimentrootpath = ""
    '''Xpath = ""
    ylabels = None'''
    
    
    def __init__(self, recordpath=metaexperimentation.expscorepath):
        # recordpath : scoreroot + annottype + setsize
        self.experimentrootpath = recordpath
        self.scorefilepath = ""     
        
        self.datapath = ""
        self.labelpath = ""
        
        self.X = None
        self.ylabels = None
        self.labelnames = None
        
        
    def train(self, datamatrix):
        return

    def test(self, originallabels):
        return
    
    def classify_hardcore(self, Xtrain, ytrain, Xtest, ytest, classifiermodel, modelname):
     
        print "...",modelname," enters:"
        
        sfit = time.time()
        print "   fit starts: "
        classifiermodel.fit(Xtrain, ytrain)        
        efit = time.time()
        fitelapsed = efit - sfit
        print " fit took ",str(fitelapsed / 60)
        
        print "   prediction starts: "                    
        ytrue, ypred = ytest, classifiermodel.predict(Xtest)
        epred = time.time()
        predelapsed = epred - efit
        print " prediction took: ",str(predelapsed / 60)
        
        self.reportresults(ytrue, ypred, modelname)
        
    
    def classify_thread(self, Xtrain, ytrain, Xtest, ytest, classifiermodel, modelname):
        p = multiprocessing.Process(target=self.classify_hardcore,
                                     kwargs={"Xtrain":Xtrain, "ytrain":ytrain,
                                             "Xtest":Xtest, "ytest":ytest,
                                             "classifiermodel":classifiermodel,
                                             "modelname":modelname})
        p.start()
        p.join(200)   # quit after 10 min.s
        if p.is_alive():
            print "Quit ",modelname
            IOtools.todisc_txt("", self.experimentrootpath+os.sep+"Quit-"+modelname+".txt")
            p.terminate()
            p.join()
        
    def classify(self, Xtrain, ytrain, Xtest, ytest, classifiermodel, modelname):
     
        print "...",modelname," enters:"
        
        print "DATA DIM ",Xtrain.shape
        
        print "   fit starts: "
        classifiermodel.fit(Xtrain, ytrain)        
                
        print "   prediction starts: "                    
        ytrue = ytest
        ypred = classifiermodel.predict(Xtest)
        
        print " ..reporting "
        self.reportresults(ytrue, ypred, modelname)
    

                
        
    def prepare_experiment(self, Xpath, ypath, erootpath, labelnames=None):
        
        self.datapath = Xpath
        self.labelpath = ypath
        
        #if erootpath:
        self.set_score_folder(erootpath)
        
        yvector = IOtools.readcsv(ypath, True)
        self.ylabels = yvector.answer.values
        yvals = self.ylabels.copy().tolist()
        #print "y vals ",yvals
        #print "vect ", self.ylabels
        if labelnames is None:
            labelnames = ["class "+str(i) for i in list(set(yvals))]

        
        instanceids = yvector.index.values.tolist()
        datadf = IOtools.readcsv(Xpath, keepindex=True)
        datadf = datadf.loc[instanceids, :]
              
        self.X = datadf.values   
        self.X[np.isnan(self.X)] = 0
        self.X[np.isinf(self.X)] = 0
        
        '''  do it inside models
        if normalize:
            self.X = preprocessing.normalize(self.X, axis=0)
        '''
        '''  can't apply standardization as it results in negative entries in the matrix, 
             which is not acceptable in Naive Bayes models but fits well in SVM.
        if standardize:
            self.X = preprocessing.scale(self.X, axis=0)
        '''
    
    
    def get_samples(self):
        return self.X, self.ylabels
    
    def set_score_folder(self, newpath):
        self.experimentrootpath = newpath
        self.scorefilepath = metaexperimentation.get_scorefilepath(self.experimentrootpath)
        IOtools.ensure_dir(os.path.join(self.experimentrootpath, "instances"))
        #self.initialize_scorefile()

    
    def reportresults(self, ytrue, ypred, experimentname):
        
        '''
        precision, recall, f1score, _ = metrics.precision_recall_fscore_support(ytrue, ypred)     
        print precision, recall, f1score
        '''
        #print ytrue
        #print ypred     
           
        precision = metrics.precision_score(ytrue, ypred, pos_label=None, average="macro")
        recall = metrics.recall_score(ytrue, ypred, pos_label=None, average="macro")
        f1score = metrics.f1_score(ytrue, ypred, pos_label=None, average="macro")
        accuracy = metrics.accuracy_score(ytrue, ypred)
        
        
        scoreline = metaexperimentation.csvsep.join(map(lambda x : str(x), [experimentname, precision, recall, f1score, accuracy]))
        IOtools.todisc_txt("\n"+scoreline, self.scorefilepath, mode="a")
        
        modelscorereportpath = os.path.join(self.experimentrootpath, experimentname+".txt")   
        try:
            scorereportstr = metrics.classification_report(ytrue, ypred, target_names=self.labelnames)
        except:
            scorereportstr = "zero division error\n"
        IOtools.todisc_txt(scorereportstr, modelscorereportpath)
        
        # record instances
        path = modelscorereportpath = os.path.join(self.experimentrootpath, "instances", experimentname+".csv")
        iheader = ["ytrue\t ypred"]
        instances = [str(true)+"\t"+str(pred) for (true, pred) in zip(ytrue, ypred)]
        IOtools.todisc_list(path, iheader+instances)
        

class SVM(LearningExperiment):
    
    methodname = "classification"
   
    kernels = []
    penalty = []
    degrees = [] 
    
    standardize = False  
    
    def __init__(self, erootpath, kernels=["rbf"], C=[1], degrees=[], standardize=False):
        LearningExperiment.__init__(self, erootpath)
        
        self.methodname = "classification"
        
        self.kernels = kernels
        self.penalty = C
        self.degrees = degrees
        self.standardize = standardize
        '''  closed to select one model  3 July 12:23
        self.kernels = ['rbf', 'poly', 'sigmoid', 'linear']   #'rbf',
        self.penalty = [1, 10, 100, 1000]
        self.degrees = range(2, 6)

        self.standardize = standardize
        '''
    
    def apply_algorithms(self, nclasses):
        
        nrows, _ = self.X.shape
        ntest = int(math.ceil(nrows * (metaexperimentation.testpercentage / 100))) 
        print "NTEST ",ntest,nrows
        Xtrain, Xtest = self.X[:-ntest, :], self.X[-ntest:, :]      
        ytrain, ytest = self.ylabels[:-ntest], self.ylabels[-ntest:] 
        
        if self.standardize:
            Xtrain = preprocessing.scale(Xtrain)
            Xtest = preprocessing.scale(Xtest)
        
        
        # train on sigmoid kernel
        clsf = SVC(kernel="sigmoid")
        modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__+"_k-sigmoid" 
        self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on rbf and linear kernels
        for k in ["rbf", "linear"]:
            for c in self.penalty:
                clsf = SVC(kernel=k, C=c)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on poly kernel
        for c in self.penalty:
            for d in self.degrees:         
                clsf = SVC(kernel="poly", C=c, degree=d)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-poly"+"_C-"+str(c)+"_d-"+str(d)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)      
                    
        
        '''
        print "apply svm"   
        for k in self.kernels:
            for c in self.penalty:
                for d in self.degrees:
                     
                    clsf = SVC(kernel=k, C=c, degree=d)
                    modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)+"_d-"+str(d)
                    #experimentname = modelname + "_nc-" + str(nclasses)
                    
                    #print ytrain,ytrain.shape
                    #print list(set(ytrain.tolist()))
                    
                    clsf.fit(Xtrain, ytrain)
                    
                    print "...",modelname
                    ytrue, ypred = ytest, clsf.predict(Xtest)
                    self.reportresults(ytrue, ypred, modelname)
        '''

    def apply_algorithms2(self, Xtrain, ytrain, Xtest, ytest):
        
        if self.standardize:
            print "standardizing the data"
            Xtrain = preprocessing.scale(Xtrain)
            Xtest = preprocessing.scale(Xtest)
            
        # no degree inclusion
        if ["poly"] not in self.kernels:
            for k in self.kernels:
                for c in self.penalty:
                    clsf = SVC(kernel=k, C=c)
                    modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)
                    self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        # continue with else if sigmoid... hope we won't need it.  3 July
        
        
        '''
        # train on sigmoid kernel
        clsf = SVC(kernel="sigmoid")
        modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__+"_k-sigmoid" 
        self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on rbf and linear kernels
        for k in ["rbf", "linear"]:
            for c in self.penalty:
                clsf = SVC(kernel=k, C=c)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on poly kernel
        for c in self.penalty:
            for d in self.degrees:         
                clsf = SVC(kernel="poly", C=c, degree=d)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-poly"+"_C-"+str(c)+"_d-"+str(d)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)      
         
        '''




class NaiveBayes(LearningExperiment):
    
    methodname = "classification"
    models = []
    
    normalize = False
    
    # selectmodels = "Gaussian-Mult"
    def __init__(self, erootpath, normalize=False, selectmodels="01"):     
        LearningExperiment.__init__(self, erootpath)
        
        self.models = []
        multinomialnb = MultinomialNB()
        gaussiannb = GaussianNB()
        if selectmodels == "01":
            self.models = [multinomialnb]
        elif selectmodels == "10":
            self.models = [gaussiannb]
        elif selectmodels == "11":
            self.models = [multinomialnb, gaussiannb]
        
        self.normalize = normalize
    
    def apply_algorithms(self, nclasses):
        
        nrows, _ = self.X.shape
        ntest = int(math.ceil(nrows * (metaexperimentation.testpercentage / 100))) 
        Xtrain, Xtest = self.X[:-ntest, :], self.X[-ntest:, :]      
        ytrain, ytest = self.ylabels[:-ntest], self.ylabels[-ntest:] 
        
        if self.normalize:
            Xtrain = preprocessing.normalize(Xtrain, axis=0)
            Xtest = preprocessing.normalize(Xtest, axis=0)
        
        '''
        multinomialnb = MultinomialNB()
        gaussiannb = GaussianNB()
        self.models = [multinomialnb, gaussiannb]
        '''
            
        print "apply naive bayes"
        for clsf in self.models:
            
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
            '''
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            clsf.fit(Xtrain, ytrain)
            print "...",modelname
            ytrue, ypred = ytest, clsf.predict(Xtest)
            self.reportresults(ytrue, ypred, modelname)
            '''

    def apply_algorithms_matrixselect(self, nclasses):
        
        nrows, _ = self.X.shape
        ntest = int(math.ceil(nrows * (metaexperimentation.testpercentage / 100))) 
        Xtrain, Xtest = self.X[:-ntest, :], self.X[-ntest:, :]      
        ytrain, ytest = self.ylabels[:-ntest], self.ylabels[-ntest:] 
        
        '''
        multinomialnb = MultinomialNB()
        gaussiannb = GaussianNB()
        self.models = [multinomialnb, gaussiannb]
        '''
        
        print "apply naive bayes"
        for clsf in self.models:
            
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
            
            

    def apply_algorithms2(self, Xtrain, ytrain, Xtest, ytest):
                
        
        #self.models = [multinomialnb, gaussiannb]
        if self.normalize:
            Xtrain = preprocessing.normalize(Xtrain, axis=0)
            Xtest = preprocessing.normalize(Xtest, axis=0)
            
        print "apply naive bayes"
        for clsf in self.models:      
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)         
             

class Clustering(LearningExperiment):
    
    models = []
    methodname = "clustering"
    
    def __init__(self, erootpath):
        LearningExperiment.__init__(self, erootpath)
        
        self.models = []
        
        self.methodname = self.__class__.__name__
    
    
    def apply_algorithms(self, nclusters):
        
        kmeans = cluster.KMeans(n_clusters=nclusters)
        #spectral = cluster.SpectralClustering(n_clusters=nclusters)
        # SPECTRAL CAUSED 'not positive definite matrix' ERROR. TOO STRICT.
        self.models = [kmeans] #, spectral]

    
        print "apply clustering"
        print len(self.models)
        i = 0
        for model in self.models:
        
            modelname = model.__class__.__name__
            experimentname = "_MT-"+self.methodname+"_alg-"+modelname   #+"_nc-"+str(nclusters)
            
            print "...",modelname
            
            #print "shape ", self.X.shape, type(nclusters)
            ytrue, ypred = self.ylabels, model.fit_predict(self.X)
            
            self.reportresults(ytrue, ypred, experimentname)
            print "CLSTR ",i
            i = i + 1


class Experimentation:
    
    datasetname = ""
    outpath = ""   # one experimentation for each datasettype (valid,setsize,features)
    scorefilepath = ""
    
    def __init__(self, experimentrootpath, datasetfolder, datasetname):
        self.datasetfolder = datasetfolder
        self.datasetname = datasetname
        
        self.outpath = experimentrootpath
        
        self.scorefilepath = os.path.join(self.outpath, "algorithms-scores.csv")   # self.datasetname+"-scores.csv")
        self.initialize_scorefile()
    
    
   
    
    def prepare_data(self, taggingtype="random"):
        print "preparing data"
    
        datamatrixcsvpath = os.path.join(self.datasetfolder, self.datasetname+".csv")
    
        print datamatrixcsvpath
        
        ylabelspath = os.path.join(self.datasetfolder, "labels", taggingtype+"-labels.csv")
        ylabels = IOtools.readcsv(csvpath=ylabelspath, keepindex=True)
        ylabels = ylabels.iloc[:, 0].values
    
        print "type(ylabels)  ",type(ylabels)
        #print ylabels
        
        return datamatrixcsvpath, ylabels
    


def conduct_cross_validation(k=5,
                             #annotationtype="double",
                             #combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             #agrtype="fullagr",
                             #labelfoldername="ALLobj-STGsubj_NC-2",
                             outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_5fold_scale/scores/"):
    
    
    annottypes = ["double", "single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
      
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("", standardize=True)
    
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, svmclassifier]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        print "sp1 ",sp1
         
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
        #agreementtypes = ["fullagr"]      
    
        feature_metric_comb_lists = utils.get_relevant_featuregroupings()  # { featureclasses : [combcode]}
                 
        for featureclass, combfilenames in feature_metric_comb_lists.iteritems():
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, featureclass))
                            
            print "metricname ",featureclass
            #print "b ",len(combfilenames)
            
            processedcombs = IOtools.getfoldernames_of_dir(sp2)
            combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
            #print "a ",len(combfilenames)
            #print "pr ",len(processedcombs),"  ",sp2
            
            
            for i,combfile in enumerate(combfilenames):
            
                print "############  ",combfile,"  ",str(i)
                Xpath = os.path.join(datasetspath, combfile + ".csv")
                sp3 = IOtools.ensure_dir(os.path.join(sp2, combfile))
                
               
                for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                    
                    lp1 = os.path.join(labelspath, agreementtype)
                    labelunions = IOtools.getfoldernames_of_dir(lp1)
                    '''labelunions = ["ALLobj-WKsubj_NC-2",
                                   "STGobj-WKsubj_NC-2",
                                   "WKobj-ALLsubj_NC-2",
                                   "WKobj-STGsubj_NC-2"]'''
                    
                    for labelunion in labelunions:
                        
                        lp2 = os.path.join(lp1, labelunion)
                        
                        labelitems = labelunion.split(metaexperimentation.interfeatsep)
                        unionname = labelitems[0]
                        ncstr = labelitems[1]
                        nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                        nc = int(nc)
                                       
                        sp4 = IOtools.ensure_dir(os.path.join(sp3, unionname))
                        
                        
                        ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                        y = IOtools.readcsv(ylabelspath, True)                
                        labelitems = y.groupby("answer").groups  # labelitems = {label : [newsid]}
                        
                        countlabels = listutils.initialize_dict(labelitems.keys(), val=0.0)
                        
                        # record test instances for guranteeing
                        testinstances = listutils.initialize_dict(labelitems.keys(), val=[])
                        traininstances = listutils.initialize_dict(labelitems.keys(), val=[])
                        for label, instanceids in labelitems.iteritems():
                            ntest = utils.get_nsplit(len(instanceids), metaexperimentation.testpercentage)
                            testinstances[label] = instanceids[-ntest:]
                            traininstances[label] = instanceids[:-ntest]
                        IOtools.todisc_json(os.path.join(sp4, "test_instances"), testinstances)
                        IOtools.todisc_json(os.path.join(sp4, "train_instances"), traininstances)
                        
                        checktrs = []
                        checktss = []
                        
                        intersectstr = []
                        intersectsts = []
                             
                        validstart = 0
                        for foldno in range(k):
                            # both will contain (fileid, label) 
                            trainitems = []  
                            testitems = []
                               
                            for label, fileids in labelitems.iteritems():
                                nvalid = utils.get_nsplit(len(fileids), metaexperimentation.validationpercentage)
                                ntest = utils.get_nsplit(len(fileids), metaexperimentation.testpercentage)
                                
                                '''
                                print " LABEL: ",label
                                print "  nvalid: ",nvalid,"  ntest: ",ntest
                                '''
                                
                                instanceids = fileids[:-ntest]
                                validstart = (foldno * (nvalid + 1)) % len(fileids)
                                validfinish = (validstart + nvalid) % len(fileids)
                                trainids = utils.gettrainset(instanceids, validstart, validfinish)  # fileids to be included in the train set
                                testids = utils.gettestset(instanceids, validstart, validfinish)  # fileids to be included in the test set
                                trainitems.extend([(fileid, label) for fileid in trainids])
                                testitems.extend([(fileid, label) for fileid in testids])

                                '''
                                print "    ntrain: ",len(trainids)
                                print "    ntestset: ",len(testids)
                                
                                
                                if len(trainids) <= len(testids):
                                    print "*******  ",foldno,labelunion, label
                                '''
                                
                                # check file collision. completed and closed 12:43
                                '''
                                coltr = listutils.getintersectionoflists(checktrs, trainids)
                                colts = listutils.getintersectionoflists(checktss, testids)
                                
                                intersectstr.extend(coltr)
                                intersectsts.extend(colts)
                                '''
                    
                            '''
                            print i," ----- ",
                            print "  intersect-train: ",intersectstr,"  ** intersect-test : ",intersectsts
                            print
                            '''
                                
                            foldpath = IOtools.ensure_dir(os.path.join(sp4, "fold-"+str(foldno)))
                            
                            metaexperimentation.initialize_score_file(foldpath)
                            
                            IOtools.tocsv_lst(trainitems, os.path.join(foldpath, "trainitems.csv"))
                            IOtools.tocsv_lst(testitems, os.path.join(foldpath, "testitems.csv"))
                            
                            Xtrain, ytrain = utils.tuple2matrix(trainitems, Xpath)
                            Xtest, ytest = utils.tuple2matrix(testitems, Xpath)
                            
                            # classify
                            for model in models:
                                model.set_score_folder(foldpath)
                                model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest)
                            
                            # random and frequency classification for baseline comparison
                            experimentname = "random"
                            distinctlabels = list(set(ytest))
                            ypred = [random.choice(distinctlabels) for _ in range(len(ytest))]
                            models[0].reportresults(ytest, ypred, experimentname)
                            
                            '''
                            experimentname = "frequency"
                            labelcount = [ytest.tolist().count(label) for label in distinctlabels]
                            ind = np.argmax(labelcount)
                            maxoccurringlabel = distinctlabels[ind]
                            ypred = [maxoccurringlabel] * len(ytest)
                            models[0].reportresults(ytest, ypred, experimentname)
                            '''
                            

# 25 Temmuz
def conduct_cross_validation_notest(k=5,
                             #annotationtype="double",
                             #combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             #agrtype="fullagr",
                             #labelfoldername="ALLobj-STGsubj_NC-2",
                             rootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/experiments_10fold_scale/scores/"):
    
    outrootpath = os.path.join(rootpath, "scores")
    
    annottypes = ["double"] #, "single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
      
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("", standardize=True)
    
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, svmclassifier]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        print "sp1 ",sp1
         
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
        #agreementtypes = ["fullagr"]      
    
        feature_metric_comb_lists = utils.get_relevant_featuregroupings()  # { featureclasses : [combcode]}
                 
        for featureclass, combfilenames in feature_metric_comb_lists.iteritems():
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, featureclass))
                            
            print "metricname ",featureclass
            #print "b ",len(combfilenames)
            
            processedcombs = IOtools.getfoldernames_of_dir(sp2)
            combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
            #print "a ",len(combfilenames)
            #print "pr ",len(processedcombs),"  ",sp2
            
            
            for i,combfile in enumerate(combfilenames):
            
                print "############  ",combfile,"  ",str(i)
                Xpath = os.path.join(datasetspath, combfile + ".csv")
                sp3 = IOtools.ensure_dir(os.path.join(sp2, combfile))
                
               
                for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                    
                    lp1 = os.path.join(labelspath, agreementtype)
                    labelunions = IOtools.getfoldernames_of_dir(lp1)
                    '''labelunions = ["ALLobj-WKsubj_NC-2",
                                   "STGobj-WKsubj_NC-2",
                                   "WKobj-ALLsubj_NC-2",
                                   "WKobj-STGsubj_NC-2"]'''
                    
                    for labelunion in labelunions:
                        
                        lp2 = os.path.join(lp1, labelunion)
                        
                        labelitems = labelunion.split(metaexperimentation.interfeatsep)
                        unionname = labelitems[0]
                        ncstr = labelitems[1]
                        nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                        nc = int(nc)
                                       
                        sp4 = IOtools.ensure_dir(os.path.join(sp3, unionname))
                        
                        
                        ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                        y = IOtools.readcsv(ylabelspath, True)                
                        labelitems = y.groupby("answer").groups  # labelitems = {label : [newsid]}
                        
                        countlabels = listutils.initialize_dict(labelitems.keys(), val=0.0)
                        
                        '''  skip test division
                        # record test instances for guranteeing
                        testinstances = listutils.initialize_dict(labelitems.keys(), val=[])
                        traininstances = listutils.initialize_dict(labelitems.keys(), val=[])
                        
                        for label, instanceids in labelitems.iteritems():
                            ntest = utils.get_nsplit(len(instanceids), metaexperimentation.testpercentage)
                            testinstances[label] = instanceids[-ntest:]
                            traininstances[label] = instanceids[:-ntest]
                        
                        IOtools.todisc_json(os.path.join(sp4, "test_instances"), testinstances)
                        IOtools.todisc_json(os.path.join(sp4, "train_instances"), traininstances)
                        '''
                        
                        checktrs = []
                        checktss = []
                        
                        intersectstr = []
                        intersectsts = []
                             
                        validstart = 0
                        for foldno in range(k):
                            # both will contain (fileid, label) 
                            trainitems = []  
                            testitems = []
                               
                            for label, fileids in labelitems.iteritems():
                                nvalid = utils.get_nsplit(len(fileids), metaexperimentation.validationpercentage)
                                #ntest = utils.get_nsplit(len(fileids), metaexperimentation.testpercentage)
                                
                                '''
                                print " LABEL: ",label
                                print "  nvalid: ",nvalid,"  ntest: ",ntest
                                '''
                                
                                instanceids = fileids   #fileids[:-ntest]
                                validstart = (foldno * (nvalid + 1)) % len(fileids)
                                validfinish = (validstart + nvalid) % len(fileids)
                                trainids = utils.gettrainset(instanceids, validstart, validfinish)  # fileids to be included in the train set
                                testids = utils.gettestset(instanceids, validstart, validfinish)  # fileids to be included in the test set
                                trainitems.extend([(fileid, label) for fileid in trainids])
                                testitems.extend([(fileid, label) for fileid in testids])

                                '''
                                print "    ntrain: ",len(trainids)
                                print "    ntestset: ",len(testids)
                                
                                
                                if len(trainids) <= len(testids):
                                    print "*******  ",foldno,labelunion, label
                                '''
                                
                                # check file collision. completed and closed 12:43
                                '''
                                coltr = listutils.getintersectionoflists(checktrs, trainids)
                                colts = listutils.getintersectionoflists(checktss, testids)
                                
                                intersectstr.extend(coltr)
                                intersectsts.extend(colts)
                                '''
                    
                            '''
                            print i," ----- ",
                            print "  intersect-train: ",intersectstr,"  ** intersect-test : ",intersectsts
                            print
                            '''
                                
                            foldpath = IOtools.ensure_dir(os.path.join(sp4, "fold-"+str(foldno)))
                            
                            metaexperimentation.initialize_score_file(foldpath)
                            
                            IOtools.tocsv_lst(trainitems, os.path.join(foldpath, "trainitems.csv"))
                            IOtools.tocsv_lst(testitems, os.path.join(foldpath, "testitems.csv"))
                            
                            Xtrain, ytrain = utils.tuple2matrix(trainitems, Xpath)
                            Xtest, ytest = utils.tuple2matrix(testitems, Xpath)
                            
                            # classify
                            
                            for model in models:
                                model.set_score_folder(foldpath)
                                model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest)
                            
                            # random and frequency classification for baseline comparison
                            experimentname = "random"
                            distinctlabels = list(set(ytest))
                            ypred = [random.choice(distinctlabels) for _ in range(len(ytest))]
                            models[0].reportresults(ytest, ypred, experimentname)
                            
                            
                            experimentname = "majority"
                            labelcount = [ytest.tolist().count(label) for label in distinctlabels]
                            ind = np.argmax(labelcount)
                            maxoccurringlabel = distinctlabels[ind]
                            ypred = [maxoccurringlabel] * len(ytest)
                            models[0].reportresults(ytest, ypred, experimentname)
                            






def get_corpus_Xpath(annotationtype, combfilename):
    
    datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
    Xpath = os.path.join(datasetspath, combfilename + ".csv")
    return Xpath
    

# outputs labelitems = {label: [fileid]}
def get_corpus_labelitems(annotationtype, agreementtype, labelunion):
    
    #labelitems = {}
    labelspath = metacorpus.get_labels_path(annotationtype)
    lp1 = os.path.join(labelspath, agreementtype)
    lp2 = os.path.join(lp1, labelunion)                 
      
    ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
    y = IOtools.readcsv(ylabelspath, True)                
    labelitems = y.groupby("answer").groups  # labelitems = {label : [newsid]}
    
    return labelitems



class corpus_definer():
    name = ""
    annotationtype = ""
    agreementtype = ""
    labelunionfoldername = ""
    size = 0
    

    def __init__(self, annottype, agrtype, lunionfolder, name):
        self.annotationtype = annottype
        self.agreementtype = agrtype
        self.labelunionfoldername = lunionfolder
        self.name = name
        self.set_size(self.get_original_size())
        print "size: ",self.size
    
    def get_Xpath(self, combfilename):
        datasetspath = metacorpus.get_datasets_path(self.annotationtype)  # finaldatasets
        Xpath = os.path.join(datasetspath, combfilename + ".csv")
        return Xpath
    
    def get_bow_Xpath(self, bow_fname):
        rawfeaturespath = os.path.join(metacorpus.learningdatapath, self.annotationtype, "rawfeatures")
        #Xpath = os.path.join(rawfeaturespath, "contenttermCOUNT.csv")
        Xpath = os.path.join(rawfeaturespath, bow_fname)
        return Xpath
    
    def get_labelitems(self):
        labelspath = metacorpus.get_labels_path(self.annotationtype)
        lp1 = os.path.join(labelspath, self.agreementtype)
        lp2 = os.path.join(lp1, self.labelunionfoldername)                 
          
        ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
        y = IOtools.readcsv(ylabelspath, True)
        
        origsize = y.shape[0]
        #print origsize,"  ->> ",self.size," <<-"
        
        if self.size != origsize:
            #print self.name,"  selecting"
            #print origsize,"  - ",self.size," -\n"
            y = y.iloc[random.sample(range(origsize), self.size), :]
            #csvpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_equaltest/items/unnnmix-"+self.name+".csv"
            #IOtools.tocsv(y, csvpath, keepindex=True)
                        
        labelitems = y.groupby("answer").groups  # labelitems = {label : [newsid]}
        
        return labelitems
    
    def get_all_labelitems(self):
        labelspath = metacorpus.get_labels_path(self.annotationtype)
        lp1 = os.path.join(labelspath, self.agreementtype)
        lp2 = os.path.join(lp1, self.labelunionfoldername)                 
          
        ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
        y = IOtools.readcsv(ylabelspath, True)                
        labelitems = y.groupby("answer").groups  # labelitems = {label : [newsid]}
        
        return labelitems
    
    
    def get_original_size(self):
        labelitems = self.get_all_labelitems()
        #numofitems = reduce(lambda x,y :len(x)+len(y), labelitems.values())
        numofitems = sum(len(i) for i in labelitems.values())
        print numofitems,"  ",type(numofitems)
        return numofitems
    
    def set_size(self, newsize):
        self.size = newsize
    
    def get_size(self):
        return self.size    
        


def charngram_classification(rootpath, 
                                charngram_csv_fname,
                                charngramXpath,
                                featureclass='redef-rat_lex-rat',
                                labelunion='ALLobj-ALLsubj_NC-2'):
    # use (super,sub)ngrams.csv as features which includes 11167 stemmed terms counted
    
    corpus = corpus_definer("double", "fullagr", labelunion, "corpus1")
    
    labelfolderitems = labelunion.split(metaexperimentation.interfeatsep)
    labelunionname = labelfolderitems[0]
    ncstr = labelfolderitems[1]
    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
    nc = int(nc)
    
    scorespath = IOtools.ensure_dir(os.path.join(rootpath, featureclass, labelunionname))
    print scorespath
    # classify
    Xpath = charngramXpath
    labelitems = corpus.get_labelitems()
         
    cross_validation_experiment(outpath=scorespath, 
                                Xtrain_path=Xpath, 
                                Xtest_path=Xpath, 
                                train_labelitems=labelitems, 
                                test_labelitems=labelitems)
        

# for comparison purposes (Ng,Dasgupta&Arifin06)
def binarypresence_classification(rootpath,                                
                                binpresXpath,
                                featureclass,
                                labelunion,
                                nfolds=10):
    # use (super,sub)ngrams.csv as features which includes 11167 stemmed terms counted
    
    corpus = corpus_definer("double", "fullagr", labelunion, "corpus1")
    
    labelfolderitems = labelunion.split(metaexperimentation.interfeatsep)
    labelunionname = labelfolderitems[0]
    ncstr = labelfolderitems[1]
    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
    nc = int(nc)
    
    scorespath = IOtools.ensure_dir(os.path.join(rootpath, featureclass, labelunionname))
    print scorespath
    # classify
    Xpath = binpresXpath
    labelitems = corpus.get_labelitems()
         
    cross_validation_experiment(outpath=scorespath, 
                                Xtrain_path=Xpath, 
                                Xtest_path=Xpath, 
                                train_labelitems=labelitems, 
                                test_labelitems=labelitems,
                                k=nfolds)



def standard_classification(rootpath,                                
                                Xpath,
                                featureclass,
                                labelunion,
                                nfolds=10):
    # use (super,sub)ngrams.csv as features which includes 11167 stemmed terms counted
    
    corpus = corpus_definer("double", "fullagr", labelunion, "corpus1")
    
    labelfolderitems = labelunion.split(metaexperimentation.interfeatsep)
    labelunionname = labelfolderitems[0]
    ncstr = labelfolderitems[1]
    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
    nc = int(nc)
    
    scorespath = IOtools.ensure_dir(os.path.join(rootpath, featureclass, labelunionname))
    print scorespath
    # classify
    labelitems = corpus.get_labelitems()
         
    cross_validation_experiment(outpath=scorespath, 
                                Xtrain_path=Xpath, 
                                Xtest_path=Xpath, 
                                train_labelitems=labelitems, 
                                test_labelitems=labelitems,
                                k=nfolds)


def bog_baseline_classification(rootpath, 
                                bow_csv_fname,
                                featureclass='redef-rat_lex-rat',
                                labelunion='ALLobj-ALLsubj_NC-2'):
    # use contenttermCOUNT.csv as features which includes 11167 stemmed terms counted
    
    corpus = corpus_definer("double", "fullagr", labelunion, "corpus1")
    
    labelfolderitems = labelunion.split(metaexperimentation.interfeatsep)
    labelunionname = labelfolderitems[0]
    ncstr = labelfolderitems[1]
    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
    nc = int(nc)
    
    scorespath = IOtools.ensure_dir(os.path.join(rootpath, featureclass, labelunionname))
    
    # classify
    Xpath = corpus.get_bow_Xpath(bow_csv_fname)
    labelitems = corpus.get_labelitems()

    #experimentname = "bogfeatures_baseline"
    #scorespath = IOtools.ensure_dir(os.path.join(outputpath, experimentname))
    
               
    cross_validation_experiment(outpath=scorespath, 
                                Xtrain_path=Xpath, 
                                Xtest_path=Xpath, 
                                train_labelitems=labelitems, 
                                test_labelitems=labelitems)
            


def experiment(rootpath, 
               featureclass='redef-rat_lex-rat',
               combfilename='comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3',
               labelunion='ALLobj-ALLsubj_NC-2'):  
    
    corpus = corpus_definer("double", "fullagr", labelunion, "corpus")
    
    labelfolderitems = labelunion.split(metaexperimentation.interfeatsep)
    labelunionname = labelfolderitems[0]
    ncstr = labelfolderitems[1]
    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
    nc = int(nc)
    
    outputpath = IOtools.ensure_dir(os.path.join(rootpath, featureclass, labelunionname))
      
    
    scorespath = IOtools.ensure_dir(os.path.join(outputpath))
    
    
    Xpath = corpus.get_Xpath(combfilename)
    labelitems = corpus.get_labelitems()
    
    
    # mix train labelitems
    trainlabelitems = labelitems.copy()
    '''for label in trainlabelitems.keys():
        l = trainlabelitems[label]
        print "before mix: ",l[:5]
        random.shuffle(l)
        trainlabelitems[label] = l
        print "after mix: ",l[:5]'''
                       
    cross_validation_experiment(outpath=scorespath, 
                                Xtrain_path=Xpath, 
                                Xtest_path=Xpath, 
                                train_labelitems=trainlabelitems, 
                                test_labelitems=labelitems)
    
    
    
            

def split_for_cross_corpus(rootpath, 
                           featureclass='redef-rat_lex-rat',
                           combfilename='comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3',
                           labelunion='ALLobj-ALLsubj_NC-2',
                           equalsize=False):
    
        
    # corpora definition
    corpus1 = corpus_definer("double", "fullagr", labelunion, "corpus1")
    corpus2 = corpus_definer("double", "halfagr", "HALFagr_NC-2", "corpus2")
    corpus3 = corpus_definer("single", "halfagr", labelunion, "corpus3")
    
    corpuslist = {1: corpus1, 2: corpus2, 3: corpus3}
    minsize = 0
    if equalsize:  # size should be set to that of the minimum-sized corpus
        sizes = []
        for corpus in corpuslist.values():
            sizes.append(corpus.get_original_size())
        minsize = min(sizes)
        for corpus in corpuslist.values():
            corpus.set_size(minsize)
    
    '''
    corpus1_Xpath = get_corpus_Xpath(corpus1.annotationtype, combfilename)
    corpus2_Xpath = get_corpus_Xpath(corpus2.annotationtype, combfilename)
    corpus3_Xpath = get_corpus_Xpath(corpus3.annotationtype, combfilename)
    
    corpus1_labelitems = get_corpus_labelitems(corpus1.annotationtype, corpus1.agreementtype, labelunion)
    corpus2_labelitems = get_corpus_labelitems(corpus2.annotationtype, corpus2.agreementtype, labelunion)
    corpus3_labelitems = get_corpus_labelitems(corpus3.annotationtype, corpus3.agreementtype, labelunion)
    '''
    
    labelfolderitems = labelunion.split(metaexperimentation.interfeatsep)
    labelunionname = labelfolderitems[0]
    ncstr = labelfolderitems[1]
    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
    nc = int(nc)
    
    outputpath = IOtools.ensure_dir(os.path.join(rootpath, featureclass, labelunionname))
    
    # experiment with tests from eah corpora
    testcase = "test_with_corpus"
    traincase = "train_corpus"
    
    for i,testset in corpuslist.iteritems():
        
        Xtestpath = testset.get_Xpath(combfilename)
        testlabelitems = testset.get_labelitems()
        
        testname = testcase + str(i)
        testfolder = IOtools.ensure_dir(os.path.join(outputpath,testname))
        
        for j,trainset in corpuslist.iteritems():
            
            Xtrainpath = trainset.get_Xpath(combfilename)
            trainlabelitems = trainset.get_labelitems()
            
            experimentname = traincase + str(j)
            
            print "\n--------\n",experimentname," ---- ",testname
            
            # ensure no intersect in train and test
            '''
            print "\n--------\n",experimentname," ---- ",testname
            intersect = True
            count = 0
           
            trainlabelitems = trainset.get_labelitems()
            trainids = []
            testids = []
            for k in trainlabelitems.keys():
                trainids.extend(trainlabelitems[k])
            for k in testlabelitems.keys():
                testids.extend(testlabelitems[k])
            if len(listutils.getintersectionoflists(trainids, testids)) != 0:
                
                count += 1
            print count,"  intersected ",len(listutils.getintersectionoflists(trainids, testids))
            print "common items: ",listutils.getintersectionoflists(trainids, testids)
            print "--------\n"
            
            '''
            
            
            scorespath = IOtools.ensure_dir(os.path.join(testfolder, experimentname))
           
                       
            cross_validation_experiment(outpath=scorespath, 
                                        Xtrain_path=Xtrainpath, 
                                        Xtest_path=Xtestpath, 
                                        train_labelitems=trainlabelitems, 
                                        test_labelitems=testlabelitems)
            
            print "--------\n"
            
            
    # train with all
    
    
    # test case for all


# (train/test)_labelitems = {label : [fileids]}
# 3 eylul
def cross_validation_experiment(outpath, Xtrain_path, Xtest_path, train_labelitems, test_labelitems, k=5):
    
    for foldno in range(k):
        # both will contain (fileid, label) 
        trainitems = []  
        testitems = []
        
        # get train items
        # for xpath in xpath_list:
        for label, fileids in train_labelitems.iteritems():
            nvalid = utils.get_nsplit(len(fileids), metaexperimentation.validationpercentage)
            # only for cross-corpus
            #nvalid = 1 
                    
            instanceids = fileids   #fileids[:-ntest]
            validstart = (foldno * (nvalid + 1)) % len(instanceids)
            validfinish = (validstart + nvalid) % len(instanceids)
            
            print "folds ",foldno
            print validstart,"  ",validfinish
            
            trainids = utils.gettrainset(instanceids, validstart, validfinish)  # fileids to be included in the train set
            
            # mix
            #random.shuffle(trainids)
            
            trainitems.extend([(fileid, label) for fileid in trainids])
            
        
        # mix
        #random.shuffle(trainitems)    
            
        # get test items                
        for label, fileids in test_labelitems.iteritems():
            nvalid = utils.get_nsplit(len(fileids), metaexperimentation.validationpercentage)
        
            
            instanceids = fileids   #fileids[:-ntest]
            validstart = (foldno * (nvalid + 1)) % len(instanceids)
            validfinish = (validstart + nvalid) % len(instanceids)
            
            testids = utils.gettestset(instanceids, validstart, validfinish)  # fileids to be included in the test set
            testitems.extend([(fileid, label) for fileid in testids])


        # ensure train and test sets do not intersect
        trainfileids = [fileid for fileid,_ in trainitems]
        testfileids = [fileid for fileid,_ in testitems]
        common = listutils.getintersectionoflists(trainfileids, testfileids) 
        print "\n",outpath
        print "  tr: ",trainfileids[:5]
        print "  ts: ",testfileids[:5]
        print "NUM OF COMMON ITEMS: ",len(common)
        print "common items: ",common
        
        
        print " ntrain before: ",len(trainitems)
        # remove colliding items from train set
        filtered_trainitems = []
        for fileid,label in trainitems:
            if fileid not in common:
                filtered_trainitems.append((fileid, label))
                
        trainitems = filtered_trainitems
        print " ntrain after: ",len(trainitems)
        
        foldpath = IOtools.ensure_dir(os.path.join(outpath, "fold-"+str(foldno)))
        
        metaexperimentation.initialize_score_file(foldpath)
        
        IOtools.tocsv_lst(trainitems, os.path.join(foldpath, "trainitems.csv"))
        IOtools.tocsv_lst(testitems, os.path.join(foldpath, "testitems.csv"))
        
       
        Xtrain, ytrain = utils.tuple2matrix(trainitems, Xtrain_path)
        Xtest, ytest = utils.tuple2matrix(testitems, Xtest_path)
        
        # classify
        run_models(foldpath, Xtrain, ytrain, Xtest, ytest)
       
    


'''
def cross_validation_experiment(outpath, Xtrain_path, Xtest_path, train_labelitems, test_labelitems, k=5):
    
    for foldno in range(k):
        # both will contain (fileid, label) 
        trainitems = []  
        testitems = []
        
        # get train items
        # for xpath in xpath_list:
        for label, fileids in train_labelitems.iteritems():
            nvalid = utils.get_nsplit(len(fileids), metaexperimentation.validationpercentage)
                    
            instanceids = fileids   #fileids[:-ntest]
            validstart = (foldno * (nvalid + 1)) % len(instanceids)
            validfinish = (validstart + nvalid) % len(instanceids)
            
            trainids = utils.gettrainset(instanceids, validstart, validfinish)  # fileids to be included in the train set
            trainitems.extend([(fileid, label) for fileid in trainids])
            
            
            
        # get test items                
        for label, fileids in test_labelitems.iteritems():
            nvalid = utils.get_nsplit(len(fileids), metaexperimentation.validationpercentage)
        
            
            instanceids = fileids   #fileids[:-ntest]
            validstart = (foldno * (nvalid + 1)) % len(instanceids)
            validfinish = (validstart + nvalid) % len(instanceids)
            
            testids = utils.gettestset(instanceids, validstart, validfinish)  # fileids to be included in the test set
            testitems.extend([(fileid, label) for fileid in testids])


        # ensure train and test sets do not intersect
        trainfileids = [fileid for fileid,_ in trainitems]
        testfileids = [fileid for fileid,_ in testitems]
        common = listutils.getintersectionoflists(trainfileids, testfileids) 
        print "\n",outpath
        print "  tr: ",trainfileids[:5]
        print "  ts: ",testfileids[:5]
        print "NUM OF COMMON ITEMS: ",len(common)
        print "common items: ",common
        
        
        print " ntrain before: ",len(trainitems)
        # remove colliding items from train set
        filtered_trainitems = []
        for fileid,label in trainitems:
            if fileid not in common:
                filtered_trainitems.append((fileid, label))
                
        trainitems = filtered_trainitems
        print " ntrain after: ",len(trainitems)
        
        foldpath = IOtools.ensure_dir(os.path.join(outpath, "fold-"+str(foldno)))
        
        metaexperimentation.initialize_score_file(foldpath)
        
        IOtools.tocsv_lst(trainitems, os.path.join(foldpath, "trainitems.csv"))
        IOtools.tocsv_lst(testitems, os.path.join(foldpath, "testitems.csv"))
        
        
        Xtrain, ytrain = utils.tuple2matrix(trainitems, Xtrain_path)
        Xtest, ytest = utils.tuple2matrix(testitems, Xtest_path)
        
        # classify
        run_models(foldpath, Xtrain, ytrain, Xtest, ytest)
'''        
      
    
def run_models(foldpath, Xtrain, ytrain, Xtest, ytest):
    
    models = []    
    svmclassifier = SVM("", standardize=True)
    nbclassifier = NaiveBayes("", normalize=True)

    models = [svmclassifier, nbclassifier]
    
    for model in models:
        model.set_score_folder(foldpath)
        model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest)
                            
    # random and frequency classification for baseline comparison
    experimentname = "random"
    distinctlabels = list(set(ytest))
    ypred = [random.choice(distinctlabels) for _ in range(len(ytest))]
    models[0].reportresults(ytest, ypred, experimentname)
    
    
    experimentname = "majority"
    labelcount = [ytest.tolist().count(label) for label in distinctlabels]
    ind = np.argmax(labelcount)
    maxoccurringlabel = distinctlabels[ind]
    ypred = [maxoccurringlabel] * len(ytest)
    models[0].reportresults(ytest, ypred, experimentname)
                            


       
                            
# final test after cross validation
def conduct_final_test(#annotationtype="double",
                             #combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             #agrtype="fullagr",
                             #labelfoldername="ALLobj-STGsubj_NC-2",
                             inrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_5fold_scale/scores/",
                             outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_final_scale/scores/"):
    
    
    

    svmclassifier = SVM("", standardize=True)
    
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, svmclassifier]
    
    
    annottypes = IOtools.getfoldernames_of_dir(inrootpath)
    
    for annottype in annottypes:
                     
            p1 = os.path.join(inrootpath, annottype)
            sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annottype))
            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            datasetspath = metacorpus.get_datasets_path(annottype)  # finaldatasets
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                sp2 = IOtools.ensure_dir(os.path.join(sp1, metricclass))
                
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                           
                for combname in featcombnames:
                    
                    Xpath = os.path.join(datasetspath, combname + ".csv")
                    
                    p3 = os.path.join(p2, combname)
                    sp3 = IOtools.ensure_dir(os.path.join(sp2, combname))
                    
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    '''labelunions = ["ALLobj-WKsubj_NC-2",
                                   "STGobj-WKsubj_NC-2",
                                   "WKobj-ALLsubj_NC-2",
                                   "WKobj-STGsubj_NC-2"]'''
                    
                    for labelunion in labelunions:
                        
                        p4 = os.path.join(p3, labelunion) 
                        sp4 = IOtools.ensure_dir(os.path.join(sp3, labelunion))
                        
                        traininstances = IOtools.readjson(os.path.join(p4, "train_instances"))
                        testinstances = IOtools.readjson(os.path.join(p4, "test_instances"))
                        # convert json to matrices
                        Xtrain, ytrain = utils.dict2matrix(traininstances, Xpath) 
                        Xtest, ytest = utils.dict2matrix(testinstances, Xpath)
                        
                        metaexperimentation.initialize_score_file(sp4)
                        for model in models:
                            model.set_score_folder(sp4)
                            model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest)
                        
                        # random and frequency classification for baseline comparison
                        experimentname = "random"
                        distinctlabels = list(set(ytest))
                        ypred = [random.choice(distinctlabels) for _ in range(len(ytest))]
                        models[0].reportresults(ytest, ypred, experimentname)
                        
                        

# final test after cross validation, excluding one feature at a time 8 July 15:24
def conduct_final_test_exludeonefeature(#annotationtype="double",
                             #combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             #agrtype="fullagr",
                             #labelfoldername="ALLobj-STGsubj_NC-2",
                             instanceidrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/experiments_5fold_scale/scores/",
                             datarootpath="/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data/",
                             outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_final_featureexclude/experiments/scores/",
                             crossvalidate=True):
    
    
    #inrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/instanceids/"

    svmclassifier = SVM("", standardize=True)
    
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, svmclassifier]
    
    
    #annottypes = IOtools.getfoldernames_of_dir(inrootpath)
    #annottypes = ["double", "single"]
    exclusiontypes = IOtools.getfoldernames_of_dir(datarootpath)
    
    for exclusionname in exclusiontypes:
    
        dp0 = os.path.join(datarootpath, exclusionname)  # root dataset path (for Xpath)
        sp0 = IOtools.ensure_dir(os.path.join(outrootpath, exclusionname))
        
        annottypes = IOtools.getfoldernames_of_dir(dp0)

        for annottype in annottypes:
                         
                ip1 = os.path.join(instanceidrootpath, annottype)
                dp1 = os.path.join(dp0, annottype)
                sp1 = IOtools.ensure_dir(os.path.join(sp0, annottype))
                
                                
                metricclasses = IOtools.getfoldernames_of_dir(dp1)
                
                #datasetspath = metacorpus.get_datasets_path(annottype)  # finaldatasets
                
                for metricclass in metricclasses:
                    
                    ip2 = os.path.join(ip1, metricclass)
                    dp2 = os.path.join(dp1, metricclass)
                    sp2 = IOtools.ensure_dir(os.path.join(sp1, metricclass))
                    
                    featcombnames = IOtools.getfoldernames_of_dir(ip2)
                    featcombnames.sort()
                    exc_featcombnames = IOtools.getfilenames_of_dir(dp2, removeextension=True)
                    exc_featcombnames.sort()
                    
                    print ip2
                    print "featcombnames: ",featcombnames
                             
                    for combname,exc_combname in zip(featcombnames, exc_featcombnames):
                        
                        Xpath = os.path.join(dp2, exc_combname + ".csv")
                        
                        ip3 = os.path.join(ip2, combname)
                        sp3 = IOtools.ensure_dir(os.path.join(sp2, exc_combname))                                                                      

                        labelunions = IOtools.getfoldernames_of_dir(ip3)
                        
                        print ip3
                        print "labelunions: ",labelunions
                        
                        for labelunion in labelunions:
                            
                            sp4 = IOtools.ensure_dir(os.path.join(sp3, labelunion))
                            ip4 = os.path.join(ip3, labelunion)
                            foldnames = IOtools.getfoldernames_of_dir(ip4)
                            
                            for foldno in foldnames:
                                
                                ip5 = os.path.join(ip4, foldno)
                                sp5 = IOtools.ensure_dir(os.path.join(sp4, foldno))
                                
                                trainitems = IOtools.csv2tuplelist(os.path.join(ip5, "trainitems.csv"))                        
                                testitems = IOtools.csv2tuplelist(os.path.join(ip5, "testitems.csv"))
                                
                                Xtrain, ytrain = utils.tuple2matrix(trainitems, Xpath)
                                Xtest, ytest = utils.tuple2matrix(testitems, Xpath)
                                
                                metaexperimentation.initialize_score_file(sp5)
                                for model in models:
                                    model.set_score_folder(sp5)
                                    model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest)
                                
                                experimentname = "random"
                                distinctlabels = list(set(ytest))
                                ypred = [random.choice(distinctlabels) for _ in range(len(ytest))]
                                models[0].reportresults(ytest, ypred, experimentname)
                                
                            



# to exclude pos and lex groups all separately  23 Temmuz 16:44
def conduct_final_test_exludeonefeature2(#annotationtype="double",
                             #combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             #agrtype="fullagr",
                             #labelfoldername="ALLobj-STGsubj_NC-2",
                             instanceidrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_5fold2/scores/",
                             datarootpath="/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data_excludegroups/groups/",
                             outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_final_featureGROUPexclude/"):
    
    
    #inrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/instanceids/"

    svmclassifier = SVM("", standardize=True)
    
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, svmclassifier]
    
    
    #annottypes = IOtools.getfoldernames_of_dir(inrootpath)
    annottypes = ["double"]
    exclusiontypes = IOtools.getfoldernames_of_dir(datarootpath)
    
    for exclusionname in exclusiontypes:
    
        dp0 = os.path.join(datarootpath, exclusionname)  # root dataset path (for Xpath)
        sp0 = IOtools.ensure_dir(os.path.join(outrootpath, exclusionname))
        

        for annottype in annottypes:
                         
                ip1 = os.path.join(instanceidrootpath, annottype)
                dp1 = os.path.join(dp0, annottype)
                sp1 = IOtools.ensure_dir(os.path.join(sp0, annottype))
                
                metricclasses = IOtools.getfoldernames_of_dir(dp1)
                
                #datasetspath = metacorpus.get_datasets_path(annottype)  # finaldatasets
                
                for metricclass in metricclasses:
                    
                    ip2 = os.path.join(ip1, metricclass)
                    dp2 = os.path.join(dp1, metricclass)
                    sp2 = IOtools.ensure_dir(os.path.join(sp1, metricclass))
                    
                    featcombnames = IOtools.getfoldernames_of_dir(ip2)
                    featcombnames.sort()
                    exc_featcombnames = IOtools.getfilenames_of_dir(dp2, removeextension=True)
                    exc_featcombnames.sort()
                    
                    print ip2
                    print "featcombnames: ",featcombnames
                             
                    for combname,exc_combname in zip(featcombnames, exc_featcombnames):
                        
                        Xpath = os.path.join(dp2, exc_combname + ".csv")
                        
                        ip3 = os.path.join(ip2, combname)
                        sp3 = IOtools.ensure_dir(os.path.join(sp2, exc_combname))
                        
                        labelunions = IOtools.getfoldernames_of_dir(ip3)
                        
                        print ip3
                        print "labelunions: ",labelunions
                        
                        for labelunion in labelunions:
                            
                            ip4 = os.path.join(ip3, labelunion) 
                            sp4 = IOtools.ensure_dir(os.path.join(sp3, labelunion))
                            
                            traininstances = IOtools.readjson(os.path.join(ip4, "train_instances"))
                            testinstances = IOtools.readjson(os.path.join(ip4, "test_instances"))
                            # convert json to matrices
                            Xtrain, ytrain = utils.dict2matrix(traininstances, Xpath) 
                            Xtest, ytest = utils.dict2matrix(testinstances, Xpath)
                            
                            metaexperimentation.initialize_score_file(sp4)
                            for model in models:
                                model.set_score_folder(sp4)
                                model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest) 
                                
                                
                                
    

def conduct_experiments2(resultspath):
    
    datafolder = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/finaldatasets_test/"
    datasetname = "feat-00111110000"
    
    datasetnames = IOtools.getfilenames_of_dir(datafolder)
    for datasetname in datasetnames: 
        epath = IOtools.ensure_dir(resultspath+os.sep+datasetname)
        experiment = Experimentation(experimentrootpath=epath, datasetfolder=datafolder, datasetname=datasetname)    
        datamatrixcsvpath, ylabels = experiment.prepare_data()
        
        #clusterer = Clustering(erootpath=epath, datamatrixpath=datamatrixcsvpath, yvector=ylabels)
        #clusterer.apply_algorithms(scorefilepath=experiment.scorefilepath)
       
        svmclassifier = SVM(erootpath=epath, datamatrixpath=datamatrixcsvpath, yvector=ylabels)
        svmclassifier.apply_algorithms(scorefilepath=experiment.scorefilepath)
        


# made 3 on 5 April
# made x on 11 May
def conduct_experimentsx(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath, scale=False):
    annottypes = ["single"]  #, "single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
    numofcombs = 15
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("")
    clusterer = Clustering("")
    nbclassifier = NaiveBayes("", normalize=scale)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [svmclassifier, nbclassifier, clusterer]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
            
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
                  
        '''  select random 
        combfilenames = IOtools.getfilenames_of_dir(datasetspath)
        combfilenames.sort()
        #combfilenames = combfilenames[:numofcombs]  #make this random
        # random select:
        #combfilenames = listutils.select_random_elements(np.array(combfilenames, dtype=object), numofcombs).tolist() 
        '''
        
        ''' process k to m 
        k = 40
        m = 130
        combfilenames = combfilenames[k:m]
        processedcombs = IOtools.getfoldernames_of_dir(sp1)
        combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
        '''
        
        ''' select previously learned '''
        '''combfilenames = IOtools.getfoldernames_of_dir(os.path.join(metacorpus.learningrootpath, 
                                                                 "experiments", 
                                                                 "scores",
                                                                 annotationtype))'''
        ''' select pregiven combs  '''
        #     for only SINGLE: 
        combfilenames = [#"comb1040_F_0-1_1-0_2-0_3-0_4-1_5-0_6-0_7-0_8-0",
                         #"comb2145_F_0-2_1-0_2-0_3-1_4-2_5-0_6-0_7-0_8-1",
                         #"comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1",
                         "comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"]
        
            
        for i,combfile in enumerate(combfilenames):
        
            print "############  ",combfile,"  ",str(i)
            Xpath = os.path.join(datasetspath, combfile + ".csv")
            sp2 = IOtools.ensure_dir(os.path.join(sp1, combfile))
            
            for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                
                lp1 = os.path.join(labelspath, agreementtype)
                labelunions = IOtools.getfoldernames_of_dir(lp1)
                
                for labelunion in labelunions:
                    
                    lp2 = os.path.join(lp1, labelunion)
                    
                    labelitems = labelunion.split(metaexperimentation.interfeatsep)
                    unionname = labelitems[0]
                    ncstr = labelitems[1]
                    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                    nc = int(nc)
                                   
                    rootscorespath = IOtools.ensure_dir(os.path.join(sp2, unionname))
                    metaexperimentation.initialize_score_file(rootscorespath)
                    ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                                    
                    for model in models:
                        
                        #labelnames = metacorpus.get_label_names()
                        model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None)
                        model.apply_algorithms(nc)
                
            print "############  ",combfile,"  ",str(i)  



# made 3 on 17 May 
def conduct_experimentsy(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath, scale=False):
    annottypes = ["single"]  #, "single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
    numofcombs = 15
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("")
    clusterer = Clustering("")
    nbclassifier = NaiveBayes("", normalize=scale)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, clusterer, svmclassifier]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        print "sp1 ",sp1
         
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
                  
    
        feature_metric_comb_lists = utils.get_featuregroupings()  # { featureclassname : { metricname : combcode-filenames } }
                 
        for featureclass, metriccombmap in feature_metric_comb_lists.iteritems():
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, featureclass))
            
            for metricname, combfilenames in metriccombmap.iteritems():
                
                sp3 = IOtools.ensure_dir(os.path.join(sp2, metricname))
                
                print "metricname ",featureclass,"  ",metricname
                print "b ",len(combfilenames)
                
                processedcombs = IOtools.getfoldernames_of_dir(sp3)
                combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
                print "a ",len(combfilenames)
                print "pr ",len(processedcombs),"  ",sp3
                
                
                for i,combfile in enumerate(combfilenames):
                
                    print "############  ",combfile,"  ",str(i)
                    Xpath = os.path.join(datasetspath, combfile + ".csv")
                    sp4 = IOtools.ensure_dir(os.path.join(sp3, combfile))
                    
                   
                    for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                        
                        lp1 = os.path.join(labelspath, agreementtype)
                        labelunions = IOtools.getfoldernames_of_dir(lp1)
                        
                        for labelunion in labelunions:
                            
                            lp2 = os.path.join(lp1, labelunion)
                            
                            labelitems = labelunion.split(metaexperimentation.interfeatsep)
                            unionname = labelitems[0]
                            ncstr = labelitems[1]
                            nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                            nc = int(nc)
                                           
                            rootscorespath = IOtools.ensure_dir(os.path.join(sp4, unionname))
                            metaexperimentation.initialize_score_file(rootscorespath)
                            ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                                            
                            for model in models:
                                print Xpath
                                #labelnames = metacorpus.get_label_names()
                                model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None)
                                model.apply_algorithms(nc)
                        
                    print "############  ",combfile,"  ",str(i)  
               
               

# made 3 on 17 May to take only 8 features
def conduct_experiments3(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath, scale=False):
    annottypes = ["single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
    numofcombs = 15
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("", standardize=True)
    clusterer = Clustering("")
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, clusterer, svmclassifier]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        print "sp1 ",sp1
         
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
                  
    
        feature_metric_comb_lists = utils.get_relevant_featuregroupings()  # { featureclasses : [combcode]}
                 
        for featureclass, combfilenames in feature_metric_comb_lists.iteritems():
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, featureclass))
                            
            print "metricname ",featureclass
            print "b ",len(combfilenames)
            
            processedcombs = IOtools.getfoldernames_of_dir(sp2)
            combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
            print "a ",len(combfilenames)
            print "pr ",len(processedcombs),"  ",sp2
            
            
            for i,combfile in enumerate(combfilenames):
            
                print "############  ",combfile,"  ",str(i)
                Xpath = os.path.join(datasetspath, combfile + ".csv")
                sp3 = IOtools.ensure_dir(os.path.join(sp2, combfile))
                
               
                for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                    
                    lp1 = os.path.join(labelspath, agreementtype)
                    labelunions = IOtools.getfoldernames_of_dir(lp1)
                    
                    for labelunion in labelunions:
                        
                        lp2 = os.path.join(lp1, labelunion)
                        
                        labelitems = labelunion.split(metaexperimentation.interfeatsep)
                        unionname = labelitems[0]
                        ncstr = labelitems[1]
                        nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                        nc = int(nc)
                                       
                        rootscorespath = IOtools.ensure_dir(os.path.join(sp3, unionname))
                        metaexperimentation.initialize_score_file(rootscorespath)
                        ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                                        
                        for model in models:
                            print Xpath
                            #labelnames = metacorpus.get_label_names()
                            model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None)
                            model.apply_algorithms(nc)
                    
                print "############  ",combfile,"  ",str(i)  
               
 

# made on 22 May
def conduct_experiments_onefeatureexcluded(inrootpath="/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data/", 
                                           outrootpath="/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/experiments/", 
                                           scale=False):
    annottypes = ["single", "double"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
    
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("", standardize=True)
    #clusterer = Clustering("")
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, svmclassifier]  # clusterer
    
    
    exclusiontypes = IOtools.getfoldernames_of_dir(inrootpath)
    
    for exclusionname in exclusiontypes:
    
        sp0 = IOtools.ensure_dir(os.path.join(outrootpath, exclusionname))
    
        for annotationtype in annottypes:
            
            sp1 = IOtools.ensure_dir(os.path.join(sp0, annotationtype))
            
            print "sp1 ",sp1
             
            datasetrootpath = os.path.join(inrootpath, exclusionname, annotationtype)  # excluded-datasets
            labelspath = metacorpus.get_labels_path(annotationtype)
            
            agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
                      
            featureclasses = IOtools.getfoldernames_of_dir(datasetrootpath)
            #feature_metric_comb_lists = utils.get_relevant_featuregroupings()  # { featureclasses : [combcode]}
                     
            for featureclass in featureclasses:
                
                sp2 = IOtools.ensure_dir(os.path.join(sp1, featureclass))
                
                dp0 = os.path.join(datasetrootpath, featureclass)
                combfilenames = IOtools.getfilenames_of_dir(dp0, removeextension=True)
                              
                print "metricname ",featureclass
                print "b ",len(combfilenames)
                
                processedcombs = IOtools.getfoldernames_of_dir(sp2)
                combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
                print "a ",len(combfilenames)
                print "pr ",len(processedcombs),"  ",sp2
                
                
                for i,combfile in enumerate(combfilenames):
                
                    print "############  ",combfile,"  ",str(i)
                    Xpath = os.path.join(dp0, combfile + ".csv")
                    sp3 = IOtools.ensure_dir(os.path.join(sp2, combfile))
                    
                   
                    for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                        
                        lp1 = os.path.join(labelspath, agreementtype)
                        labelunions = IOtools.getfoldernames_of_dir(lp1)
                        
                        for labelunion in labelunions:
                            
                            lp2 = os.path.join(lp1, labelunion)
                            
                            labelitems = labelunion.split(metaexperimentation.interfeatsep)
                            unionname = labelitems[0]
                            ncstr = labelitems[1]
                            nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                            nc = int(nc)
                                           
                            rootscorespath = IOtools.ensure_dir(os.path.join(sp3, unionname))
                            metaexperimentation.initialize_score_file(rootscorespath)
                            ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                                            
                            for model in models:
                                print Xpath
                                #labelnames = metacorpus.get_label_names()
                                model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None)
                                model.apply_algorithms(nc)
                        
                    print "############  ",combfile,"  ",str(i)  







           
                   
    
def conduct_experiments(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath, normalize=False):
    annottypes = ["double"]
    setsizes = ["150"]
    taggertypes = ["user"]
    numofcombs = 5
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("")
    clusterer = Clustering("")
    nbclassifier = NaiveBayes("")
    #nbclassifier = MultinomialNB(outrootpath)
    models = [svmclassifier, nbclassifier, clusterer]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        for setsize in setsizes:
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, setsize))
            
            datasetspath = metacorpus.get_datasets_path(annotationtype, setsize)  # finaldatasets
            labelspath = metacorpus.get_labels_path(annotationtype, setsize)
            nclasses = IOtools.getfoldernames_of_dir(labelspath)
                      
            combfilenames = IOtools.getfilenames_of_dir(datasetspath)
            combfilenames = combfilenames[:numofcombs]
            
            for combfile in combfilenames:
            
                Xpath = os.path.join(datasetspath, combfile + ".csv")
                sp3 = IOtools.ensure_dir(os.path.join(sp2, combfile))
                
                for nclass in nclasses:   # count it on labelspath not nclasses
                    
                    #nclabelspath = arrange_N_classes.nclass_label_folder(labelspath, nc)  # get folder path containing nc-grouped labels
                    nclabelspath = os.path.join(labelspath, nclass)
                    nc = nclass.split(metaexperimentation.intrafeatsep)[-1]
                    nc = int(nc)
                    sp4 = IOtools.ensure_dir(os.path.join(sp3, nclass)) #"NC-"+str(nc)))
                    
                    for taggertype in taggertypes:
                        
                        rootscorespath = IOtools.ensure_dir(os.path.join(sp4, taggertype))
                        metaexperimentation.initialize_score_file(rootscorespath)
                        ylabelspath = os.path.join(nclabelspath, taggertype+".csv")
                        
                        for model in models:
                            
                            #labelnames = metacorpus.get_label_names()
                            model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None, normalize=normalize)
                            model.apply_algorithms(nc)    
            

def evaluate_performance(resultspath):
    performancepath = os.path.join(resultspath, "performance")  # should be taken from meta experimentation
    datasetcombs = IOtools.getfoldernames_of_dir(os.path.join(resultspath, "scores"))
    
    
    
    
    # best comb per alg
    # best alg per comb
    # worst alg per comb_excld1or2
    # best and worst first 10 settings
    
    # interpreting comb names is important
    # parsing mt and alg names seems easier


def many_types_exclude():
    
    '''  closed on 28 Temmuz
    rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/ablation/"
    instanceidrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_5fold_scale/scores/"
    exclusiontypes = {"group" : "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data_excludegroups/groups/",
                      "item" : "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data/",
                      "onedim" : "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data_excludegroups/onedim/"}
    
    for outfoldername, datapath in exclusiontypes.iteritems():
        outrootpath = os.path.join(rootpath, outfoldername)
        scorespath = IOtools.ensure_dir(os.path.join(outrootpath, "scores"))
        conduct_final_test_exludeonefeature(instanceidrootpath=instanceidrootpath, datarootpath=datapath, outrootpath=scorespath)
        performance_evaluation2.evaluate_featureexcluded_datasets2(outrootpath)
        '''
    rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/ablation2/"
    instanceidrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/experiments_5fold_scale/scores/"
    exclusiontypes = {"group" : "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data_excludegroups/groups/",
                      "item" : "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data/",
                      "onedim" : "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/data_excludegroups/onedim/"}
    
    for outfoldername, datapath in exclusiontypes.iteritems():
        outrootpath = os.path.join(rootpath, outfoldername)
        scorespath = IOtools.ensure_dir(os.path.join(outrootpath, "scores"))
        conduct_final_test_exludeonefeature(instanceidrootpath=instanceidrootpath, datarootpath=datapath, outrootpath=scorespath)
        #for exctype in IOtools.getfoldernames_of_dir(outrootpath):
            #performance_evaluation_crossval.evaluate_cross_validation(os.path.join(outrootpath, exctype))
        #performance_evaluation2.evaluate_featureexcluded_datasets2(outrootpath)


# per datasetname  (random/size)
def shell():
    #resultspath = metaexperimentation.rootpath + "/test-N5/"
    
    #conduct_cross_validation_notest()
    #conduct_cross_validation()   # 3 July 11:14
    #conduct_final_test()
    '''conduct_final_test(inrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/exp-weaklabels/crossval/",
                       outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning10/exp-weaklabels/finaltest/")
    '''
    #many_types_exclude()
    
    #conduct_final_test_exludeonefeature2()  # 8 July 15:41

    #conduct_experiments3(scale=True)
    #evaluate_performance(resultspath)
    #conduct_experiments_onefeatureexcluded()
    
    
    # 3 Eylul
    #conduct_cross_validation_notest(outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/5fold_test20p/scores/")
    
    '''
    split_for_cross_corpus(rootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_equaltest/items",
                           equalsize=True)
    '''
    split_for_cross_corpus(rootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_test_collision/UNequalsize",
                           equalsize=False)
    #bog_baseline_classification(rootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/bogbaseline")
   
   
if __name__ == "__main__":
    shell()
    #conduct_experiments_onefeatureexcluded()
    
    
    
    
    
    
    
    

