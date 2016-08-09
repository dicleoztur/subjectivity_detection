'''
Created on Jul 28, 2014

@author: dicle
'''

from nltk import ConditionalFreqDist


def subword_char_ngram(text_fileid_map, n):
    corpus_ngramitems = []
    for tid, text in text_fileid_map.iteritems():
        words = text.split()
        ngramitems = []
        for w in words:
            ngramitems.extend(ngrams(w, n))
        for ngramitem in ngramitems:
            corpus_ngramitems.append((tid, ngramitem))
    cfd = ConditionalFreqDist(corpus_ngramitems)
    print cfd.N()," ",len(cfd.conditions())
    return cfd
    CFDhelpers.cfd2csv(cfd=cfd, csvpath=csvpath)

def conduct_cross_validation_notest(k=10,
                             #annotationtype="double",
                             #combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             #agrtype="fullagr",
                             #labelfoldername="ALLobj-STGsubj_NC-2",
                             outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/experiments_10fold_scale/scores/"):
    
    
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
                                                      
                                instanceids = fileids   #fileids[:-ntest]
                                validstart = (foldno * (nvalid + 1)) % len(fileids)
                                validfinish = (validstart + nvalid) % len(fileids)
                                trainids = utils.gettrainset(instanceids, validstart, validfinish)  # fileids to be included in the train set
                                testids = utils.gettestset(instanceids, validstart, validfinish)  # fileids to be included in the test set
                                trainitems.extend([(fileid, label) for fileid in trainids])
                                testitems.extend([(fileid, label) for fileid in testids])

                                
                                
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
                            

     