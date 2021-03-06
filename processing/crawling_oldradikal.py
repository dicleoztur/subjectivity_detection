# -*- coding: utf-8 -*- 
'''
Created on Dec 26, 2013

@author: dicle
'''


'''
Created on Aug 6, 2012

@author: dicle
'''

'''
radikal değişmiş, o yüzden eski halini buraya yedekledim. 
'''

import urllib2
from urllib2 import URLError
import re
import os
import nltk
import time
import random

from NewsItem import NewsItem
from NewsResource import NewsResource
#from IO import *
import IO


cat_radikal = { 81:"dunya", 77:"turkiye", 80:"ekonomi", 78:"politika", 84:"spor" }     
#cat_radikal = { 41:"hayat", 80:"ekonomi" }   #sinema: 120 bitti.   
#cat_radikal = { 78: "politika"}

#cat_habervaktim = { 3:"guncel", 4:"siyaset", 5:"dunya", 6:"ekonomi", 7:"kultur-sanat", 8:"aile-yasam", 10:"bilim", 11:"saglik", 19:"spor", 20:"egitim", 23:"medya" }
#cat_habervaktim = {  11:"saglik" } #, 6:"ekonomi"}
cat_habervaktim = { 3:"guncel", 4:"siyaset", 5:"dunya", 6:"ekonomi", 23:"medya"} #, 11:"saglik"  }

#cat_cumhuriyet = {6:"siyaset", 7:"turkiye", 8:"dunya", 9:"ekonomi", 12:"kultur-sanat"} #, 17:"spor", 20:"yasam", 18:"bilim-teknik", 19:"saglik", 21:"cevre" }
cat_cumhuriyet = {7:"turkiye" }

def extractitem(marker1, marker2, rawhtml):    #, printt = False):
    start = rawhtml.find(marker1)
    
    if start == -1:
        item = ''
    else:
        start2 = start+len(marker1)
        end = rawhtml.find(marker2, start2)
        item = rawhtml[start2 : end]
    
    '''end = rawhtml.find(marker2)
    
    if start == -1:
        item = ''
    else:
        item = rawhtml[start+len(marker1) : end]
        '''
        
    '''
        if printt:
            print rawhtml[start : start+len(marker1)]
            print rawhtml[end-10 : end+len(marker2)+10]
            print "start: ",start
            print "scope: ",str(end-start-len(marker1))
            print "end: ",end
            print "len: ",str(len(item))
            '''
    return item



def getnewsitem(resource, url, newsid):
    '''
    f = urllib2.urlopen(url)
    rawhtml = f.read()
    #rawhtml = rawhtml.encode('iso-8859-9')
    f.close()
    '''
    
    req = urllib2.Request(url)
    try:
        f = urllib2.urlopen(req)
    except URLError as e:
        message = ""
        if hasattr(e, 'reason'):
            message += 'Cannot reach a server.'
            message += '\nReason: ' + str(e.reason)
        elif hasattr(e, 'code'):
            message += 'The server couldn\'t fulfill the request.'
            message += '\nError code: ' + str(e.code)
        IO.log_connection_error(resource.name, url, message)
        return None
    else:
        rawhtml = f.read()
        #rawhtml = rawhtml.encode('iso-8859-9')
        f.close()
    
    
    encoding = resource.encoding
    if encoding == "":
        encoding = f.headers['content-type'].split('charset=')[-1]
        resource.setEncoding(encoding)
    
    markerTitle1 = resource.markerTitle1
    markerTitle2 = resource.markerTitle2
    title = extractitem(markerTitle1, markerTitle2, rawhtml)
    title = IO.encodingToutf8(title, encoding)    
    title = re.split(r"[/-]", title)[0]
    title = IO.replaceSpecialChars(title)  
    
    markerDate1 = resource.markerDate1 
    markerDate2 = resource.markerDate2
    date = extractitem(markerDate1, markerDate2, rawhtml)
    date = IO.encodingToutf8(date, encoding)
    #date = nltk.clean_html(date)
    
    #print "markers: ",markerDate1," ",markerDate2
    print "date: ",date
    
    
    markerAuthor1 = resource.markerAuthor1
    markerAuthor2 = resource.markerAuthor2
    author = extractitem(markerAuthor1, markerAuthor2, rawhtml)
    
    markerText1 = resource.markerText1
    markerText2 = resource.markerText2
    text = extractitem(markerText1, markerText2, rawhtml)
    
    print isinstance(text, str)," ",isinstance(text, unicode)," ",type(text)
    text = IO.encodingToutf8(text, encoding)
    print isinstance(text, str)," ",isinstance(text, unicode)," ",type(text)
    text = nltk.clean_html(text)
    
    
    
    '''
    print isinstance(text, str)," ",isinstance(text, unicode)," ",type(text)
    text = text.decode('utf-8', 'ignore')
    print isinstance(text, str)," ",isinstance(text, unicode)," ",type(text)
    text = nltk.clean_html(text)
    
    text = IO.encodingToutf8(text, encoding)
    '''
    text = IO.replaceSpecialChars(text)
    
    return NewsItem(newsid, title, date, text, resource.name, author, url)


def retrieveNewsIDs(resource, link):    
    content = readhtml(link)
    pattern1 = resource.idpatternInLink1
    pattern2 = resource.idpatternInLink2
        
    #preprocess to get the raw list of news links  
    
    limit1 = resource.idlimit1
    limit2 = resource.idlimit2
    item = extractitem(limit1,limit2,content)
    
    # extract ids
    list1 = re.findall(pattern1,item)
    #print list1
    listfinal = []
    i = 0
    for rawID in list1:
        newsid = re.findall(pattern2, rawID)[0]      #buradan bug cikabilir
        listfinal.append(newsid)
        #print str(i)," - ",newsid
        i = i+1

    listfinal = list(set(listfinal))
    return listfinal



def readhtml(url):
    f = urllib2.urlopen(url)
    rawhtml = f.read()
    #rawhtml = rawhtml.encode('iso-8859-9')
    f.close()
    
    return rawhtml



# crawl the newsitem links in the resource (on the dunya pages), record the newsites ids and thus their links, read the items, record the texts
# start and finish denote the time and page interval to crawl the ids.
def crawlresourceItems(resource, IDlist, categoryname):
    
    path = IO.ensure_dir(IO.itemsPath+os.sep+resource.name+os.sep+categoryname+os.sep)
    rootlink_news = resource.rootlink_item
    for newsid in IDlist:
        newslink = rootlink_news + str(newsid)
        if resource.name == "vakit":
            newslink += "/"
        print newslink
        extraction = getnewsitem(resource, newslink, newsid)
        if extraction:
            extraction.setcategory(categoryname)
            time.sleep(random.choice(range(3,10)))    #time.sleep(20)
            #extraction.toConsole()
            extraction.toDisc(path)


# should take (startdate, enddate)
def crawl_habervaktim(start, numOfPages, categoryID):
    name = "vakit"
    catname = cat_habervaktim[categoryID]
    rootlink_item = "http://www.habervaktim.com/haber/"   # ex. http://www.habervaktim.com/haber/316875/
    rootlink_id = "http://www.habervaktim.com/"+ catname +"-haberleri-"+ str(categoryID) +"hk-p"    # ex. http://www.habervaktim.com/siyaset-haberleri-4hk-p5.htm
    
    #item
    markerTitle1 = '<div class="title"><h1>'
    markerTitle2 = '</h1></div>'
    
    markerText1 = 'class="text_content">'
    markerText2 = 'changeTarget("#news_content")'
    
    markerDate1 = '<div class="date">'
    markerDate2 = '<div id="news_content"'
    
    markerAuthor1 = ""
    markerAuthor2 = ""
    
    idlimit1 = '<div class="news"><div class="box_news box_news_1">'
    idlimit2 = '<div class="hor_seperator">'
    
    pattern1 = r"/haber/[0-9]{6,9}"
    pattern2 = r'[0-9]{6,9}'    
    
    resource1 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2, markerDate1, markerDate2, markerAuthor1, markerAuthor2)
    resource1.setEncoding('UTF-8')
    
    rooturl = resource1.rootlink_id
    IDlist = []
    for i in range(start,start+numOfPages):
        url = rooturl + str(i) + ".htm"
        IDlist = IDlist + retrieveNewsIDs(resource1, url)
    
    IDlist = list(set(IDlist))
    categoryName = cat_habervaktim[categoryID]
    path = resource1.newsidpath+categoryName+"_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
    IO.todisc_list(IDlist, path)
    
    crawlresourceItems(resource1, IDlist, categoryName)
    

# Take date range as input
def crawl_radikal(start, numOfPages, categoryID):
    
    name = "radikal"
    rootlink_item = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID="
    #rootlink_id = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID=81&PAGE="
    #rootlink_id = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID="+str(categoryID)+"&PAGE="
    rootlink_id = "http://www.radikal.com.tr/"+str(cat_radikal[categoryID])+"/tum_haberler-"
    
    #item
    markerTitle1 = '<title>'
    markerTitle2 = '</title>'
    
    
    '''  eski:
    markerText1 = '<div id="metin2" class="fck_li">'
    markerText2 = '<div class="IndexKeywordsHeader"'    # veya 'id="hiddenTitle"'
    
    idlimit1 = "<div class=\"cat-news\"><ol";
    idlimit2 = "var Geri = 'Geri'";
    
    '''
    
    ''' 10 Ekim itibariyle  '''
    markerText1 = '<div id="metin2" class="article"'
    markerText2 = '<div class="article_end"'
    
    markerDate1 = '<div class="text_size"><span>'   #'<p class="date">'
    markerDate2 = '</span><span>'  #'</p>'
    
    markerAuthor1 = '=MuhabirArama&amp;Keyword='
    markerAuthor2 = '</a>'
    
    idlimit1 = "<div class=\"box_z_a\"";
    idlimit2 = "<div id=\"paging\"";
    
    pattern1 = r";articleid=[0-9]{6,9}"    #r";ArticleID=[0-9]{6,9}"
    pattern2 = r'[0-9]{6,9}'
    
    
    resource1 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2, markerDate1, markerDate2, markerAuthor1, markerAuthor2)
    resource1.setEncoding('iso-8859-9')
    
    #start = 1
    #numOfPages = 2
    rooturl = resource1.rootlink_id
    IDlist = []
    for i in range(start,start+numOfPages):
        url = rooturl + str(i)
        IDlist = IDlist + retrieveNewsIDs(resource1, url)
    
    IDlist = list(set(IDlist))
    categoryName = cat_radikal[categoryID]
    path = resource1.newsidpath+categoryName+"_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
    IO.todisc_list(IDlist, path)
    
    crawlresourceItems(resource1, IDlist, categoryName)
    



# for unicode problem catching
def crawl_radikal2(categoryID, IDlist):
    
    name = "radikal"
    rootlink_item = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID="
    #rootlink_id = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID=81&PAGE="
    rootlink_id = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID="+str(categoryID)+"&PAGE="
    
    #item
    markerTitle1 = '<title>'
    markerTitle2 = '</title>'
    
    
    '''  eski:
    markerText1 = '<div id="metin2" class="fck_li">'
    markerText2 = '<div class="IndexKeywordsHeader"'    # veya 'id="hiddenTitle"'
    
    idlimit1 = "<div class=\"cat-news\"><ol";
    idlimit2 = "var Geri = 'Geri'";
    
    '''
    
    ''' 10 Ekim itibariyle  '''
    markerText1 = '<div id="metin2" class="article"'
    markerText2 = '<div class="article_end"'
    
    markerDate1 = '<p class="date">'
    markerDate2 = '</p>'
    
    markerAuthor1 = '=MuhabirArama&amp;Keyword='
    markerAuthor2 = '</a>'
    
    idlimit1 = "<div class=\"box_z_a\"";
    idlimit2 = "<div id=\"paging\"";
    
    pattern1 = r";ArticleID=[0-9]{6,9}"
    pattern2 = r'[0-9]{6,9}'
    
    
    resource1 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2, markerDate1, markerDate2, markerAuthor1, markerAuthor2)
    resource1.setEncoding('iso-8859-9')
    

    
    IDlist = list(set(IDlist))
    categoryName = cat_radikal[categoryID]
    
    
    crawlresourceItems(resource1, IDlist, categoryName)
    
    

def crawl_cumhuriyet(start, numOfPages, categoryID):
    
    name = "cumhuriyet"
    rootlink_item = "http://www.cumhuriyet.com.tr/?hn="
    rootlink_id = "http://www.cumhuriyet.com.tr/?kn="+ str(categoryID) + "&ilk="
    
    
    
    '''  eski
    markerText1 = 'data-text="'                   #'class="twitter-share-button"'     #'<span class="mahrec">'
    markerText2 = '<p class="tarih">'    # veya 'id="hiddenTitle"'
    '''
    
    #item
    markerTitle1 = '<span class="ehaberBaslik">'
    markerTitle2 = '</span></h1>'
    
    markerText1 = 'class="mahrec"'                   #'class="twitter-share-button"'     #'<span class="mahrec">'
    markerText2 = '<p class="tarih">'    # veya 'id="hiddenTitle"'
    
    
    
    
    '''  11 Ekim itibariyle '''
    
    markerDate1 = '<p class="tarih">'
    markerDate2 = '</p>'      #'<form id="FORMyildiz">'
    
    markerAuthor1 = '?yer=yazar&amp;aranan='
    markerAuthor2 = '">'
    
    
    #id
    idlimit1 = '<div class="s1st1_1">'
    idlimit2 = '<div class="s1st1_2">'
    
    pattern1 = r"\?hn=[0-9]{5,9}"
    pattern2 = r'[0-9]{5,9}'
    
    
    resource = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2, markerDate1, markerDate2, markerAuthor1, markerAuthor2)
    resource.setEncoding('iso-8859-9')
    
    
    
    
    # get id list
    
    #start = 0
    #numOfPages = 2
    rooturl = resource.rootlink_id
    IDlist = []
    for i in range(start,start+numOfPages):
        url = rooturl + str(i*15)
        print url
        IDlist = IDlist + retrieveNewsIDs(resource, url)
    
    IDlist = list(set(IDlist))
    categoryName = cat_cumhuriyet[categoryID]
    path = resource.newsidpath+categoryName+"_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
    IO.todisc_list(IDlist, path)
    
    crawlresourceItems(resource, IDlist, categoryName)




def crawl_yenisafak(start, numOfPages):   
    name = "yenisafak"
    rootlink_item = "http://www.yenisafak.com.tr/Dunya/?i="
    rootlink_id = "http://www.yenisafak.com.tr/Dunya/"    # http://www.yenisafak.com.tr/Dunya/?t=dd.mm.yyyy
    
    
    #item
    markerTitle1 = '<title>'
    markerTitle2 = '</title>'
    
    markerText1 = 'class="haberdetaymetin">'                   
    markerText2 = 'class="haberdetaytarih"'    
    
    #id
    idlimit1 = '<div class="haberdetaydiger2';
    idlimit2 = "<div class=\"mngalerivideo";
    
    pattern1 = r"\&i=[0-9]{5,9}"
    pattern2 = r'[0-9]{5,9}'
    
    resource3 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2)
    
    
    
    # get id list
    
    #start = 0
    #numOfPages = 2
    rooturl = resource3.rootlink_id
    IDlist = []
    #for one day:
    IDlist = IDlist + retrieveNewsIDs(resource3, rooturl)
    
    IDlist = list(set(IDlist))
    path = resource3.newsidpath+"dunya_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
    IO.todisc_list(IDlist, path)   
        
    
    crawlresourceItems(resource3, IDlist)


    


#############    MAIN    #####################




'''
start = 0
numOfPages = 0

crawl_radikal(start, numOfPages)
'''
    

if __name__ == "__main__":
    
    jump = 2   # jump ten pages at each cycle to have a somewhat more uniform time range of news
    numOfPages = 25  # 90
    steps = 20
    initialize=1
    start = 40
    
    
    #crawl_radikal2(78, ['1115404'])
    
    
    s = 1     # bu haliyle 26Kasim dan basladi. onlem al; biraz daha ilerden haber ceksin. 
    steps = 10
    start = 15
    for cid, cname in cat_radikal.iteritems():
    #for cid, cname in cat_habervaktim.iteritems():
    #for cid, cname in cat_cumhuriyet.iteritems(): 
        #crawl_cumhuriyet(start, numOfPages, cid)
        crawl_radikal(start, numOfPages, cid)
        #crawl_habervaktim(start, numOfPages, cid)
    
    
    
    
    ''' close on 2 Nisan for getting all the news
    #for cid, cname in cat_cumhuriyet.iteritems():
    for cid, cname in cat_habervaktim.iteritems():
    #for cid, cname in cat_radikal.iteritems():
        for i in range(initialize,initialize+steps):
            #numOfPages = 
            #crawl_radikal(start, numOfPages, cid)
            crawl_habervaktim(start, numOfPages, cid)
            start += 2*i
            #crawl_cumhuriyet(start, numOfPages, cid)
    '''


''' 
NOTLAR
id retrieval icin date'e gore range belirleme kismi eksik. Bu haliyle en guncel, son iki veya bir gunun haber id'leri cekiliyor
'''

'''   start with initializing radikal as a NewsResource object   '''


'''
name = "radikal"
rootlink_item = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID="
rootlink_id = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID=81&PAGE="


#item
markerTitle1 = '<title>'
markerTitle2 = '</title>'

markerText1 = '<div id="metin2" class="fck_li">'
markerText2 = '<div class="IndexKeywordsHeader"'    # veya 'id="hiddenTitle"'


#id
idlimit1 = "<div class=\"cat-news\"><ol";
idlimit2 = "var Geri = 'Geri'";

pattern1 = r";ArticleID=[0-9]{6,9}"
pattern2 = r'[0-9]{6,9}'


resource1 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2)




start = 1
numOfPages = 2
rooturl = resource1.rootlink_id
IDlist = []
for i in range(start,start+numOfPages):
    url = rooturl + str(i)
    IDlist = IDlist + retrieveNewsIDs(resource1, url)

IDlist = list(set(IDlist))
path = resource1.newsidpath+"dunya_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
IO.todisc_list(IDlist, path)


crawlresourceItems(resource1, IDlist)
'''





'''

rootlink_news = resource1.rootlink_item
for newsid in IDlist:
    newslink = rootlink_news + str(newsid)
    extraction = getnewsitem(resource1, newslink)
    extraction.toConsole()
    
  
 
    '''







'''   cumhuriyet    '''
    

'''
name = "cumhuriyet"
rootlink_item = "http://www.cumhuriyet.com.tr/?hn="
rootlink_id = "http://www.cumhuriyet.com.tr/?kn=8&ilk="


#item
markerTitle1 = '<span class="ehaberBaslik">'
markerTitle2 = '</span></h1>'

markerText1 = 'data-text="'                   #'class="twitter-share-button"'     #'<span class="mahrec">'
markerText2 = '<p class="tarih">'    # veya 'id="hiddenTitle"'


#id
idlimit1 = "<div class=\"s1st1_1\">";
idlimit2 = "<div class=\"s1st1_2\">";

pattern1 = r"\?hn=[0-9]{5,9}"
pattern2 = r'[0-9]{5,9}'


resource2 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2)
resource2.setEncoding('iso-8859-9')




# get id list

start = 0
numOfPages = 2
rooturl = resource2.rootlink_id
IDlist = []
for i in range(start,start+numOfPages):
    url = rooturl + str(i*15)
    print url
    IDlist = IDlist + retrieveNewsIDs(resource2, url)

IDlist = list(set(IDlist))
path = resource2.newsidpath+"dunya_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
IO.todisc_list(IDlist, path)


crawlresourceItems(resource2, IDlist)


'''




'''
# get news items


rootlink_news = resource1.rootlink_item
for newsid in IDlist:
    newslink = rootlink_news + str(newsid)
    extraction = getnewsitem(resource1, newslink)
    extraction.toConsole()
    
''' 













'''  yeni safak    '''
    
'''
name = "yenisafak"
rootlink_item = "http://www.yenisafak.com.tr/Dunya/?i="
rootlink_id = "http://www.yenisafak.com.tr/Dunya/"    # http://www.yenisafak.com.tr/Dunya/?t=dd.mm.yyyy


#item
markerTitle1 = '<title>'
markerTitle2 = '</title>'

markerText1 = 'class="haberdetaymetin">'                   
markerText2 = 'class="haberdetaytarih"'    

#id
idlimit1 = '<div class="haberdetaydiger2';
idlimit2 = "<div class=\"mngalerivideo";

pattern1 = r"\&i=[0-9]{5,9}"
pattern2 = r'[0-9]{5,9}'

resource3 = NewsResource(name, rootlink_id, rootlink_item, idlimit1, idlimit2, pattern1, pattern2, markerTitle1, markerTitle2, markerText1, markerText2)



# get id list

start = 0
numOfPages = 2
rooturl = resource3.rootlink_id
IDlist = []
#for one day:
IDlist = IDlist + retrieveNewsIDs(resource3, rooturl)

IDlist = list(set(IDlist))
path = resource3.newsidpath+"dunya_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
IO.todisc_list(IDlist, path)   
    

crawlresourceItems(resource3, IDlist)

'''


   
# get news items



'''
i=0

path = IO.ensure_dir(IO.itemsPath+os.sep+resource1.name+os.sep)
rootlink_news = resource1.rootlink_item
for newsid in IDlist:
    newslink = rootlink_news + str(newsid)
    extraction = getnewsitem(resource1, newslink, newsid)
    extraction.toConsole()
    
    print "PATH, ",path
    extraction.toDisc(path)
    i = i+1
    print str(i)
    
    '''
    