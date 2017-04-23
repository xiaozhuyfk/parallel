'''
Created on Dec 3, 2015 2:19:48 PM
@author: cx

what I do:
    i fetch the text and made lm from freebase for target ids
what's my input:

what's my output:


'''

import site
site.addsitedir('/bos/usr0/cx/PyCode/ExplicitSemanticEncoder')

from FreebaseDumpParser import FreebaseDumpParserC
from FreebaseDumpReader import FreebaseDumpReaderC
#from EseTextBase import EseLmC
import ConfigParser
#import json
from util import writeFile


lTargetField = ['Name','Desp','Alias']




def Process(DumpInName,TargetIdIn,OutPre):

    reader = FreebaseDumpReaderC()
    reader.open(DumpInName)
    Parser = FreebaseDumpParserC()
    global lTargetField

    sTargetId = set([item.split('\t')[0] for item in open(TargetIdIn).read().splitlines()])

    #lOut = [open(OutPre + '_' + field, 'w') for field in lTargetField]

    for cnt,lvCol in enumerate(reader):

        if 0 == (cnt % 1000):
            print 'read [%d] obj' %(cnt)

        ObjId = Parser.GetObjId(lvCol)
        if not ObjId in sTargetId:
            continue

        #lText = [Parser.GetField(lvCol, field) for field in lTargetField]
        #lLm = [EseLmC(text) for text in lText]

        #for out, lm in zip(lOut,lLm):
        #   print >>out, ObjId + '\t' + json.dumps(lm.hTerm)


    #for out in lOut:
    #    out.close()

    print 'finished'
    return


import sys

"""
ConfSec = 'EntityLmFromDump'
if 2 != len(sys.argv):
    print 'I simply fetch field texts for entity in Freebase'
    print '1 para, conf'
    print '[%s]' %(ConfSec)
    print 'DumpIn=\nTargetId=\nOut=\n'
    sys.exit()

conf = ConfigParser.SafeConfigParser()
conf.read(sys.argv[1])

DumpIn = conf.get(ConfSec,'DumpIn')
TargetId = conf.get(ConfSec,'TargetId')
OutPre = conf.get(ConfSec,'Out')

Process(DumpIn, TargetId, OutPre)
"""

edges = [
    #"http://rdf.freebase.com/ns/astronomy.astronomical_discovery.discovery_technique",
    "<http://rdf.freebase.com/key/wikipedia.en_id>"
]

result_file = "../testresult/dump/pairs"

def extract_wikiurl():
    dump = '/home/hongyul/AMA/freebase_dump/freebase-rdf-latest.gz'
    result_path = '/home/hongyul/AMA/freebase_dump/wikiurl'
    writeFile(result_path, "")
    reader = FreebaseDumpReaderC()
    reader.open(dump)
    Parser = FreebaseDumpParserC()
    d = {}

    for cnt,lvCol in enumerate(reader):

        if 0 == (cnt % 1000):
            print 'read [%d] obj' %(cnt)

        ObjId = Parser.GetObjId(lvCol)
        wiki_url = Parser.GetWikiUrl(lvCol)

        if ObjId and wiki_url:
            content = "\t".join([ObjId] + wiki_url) + "\n"
            writeFile(result_path, content, "a")

    print 'finished'
    return

def test():
    file = "/home/hongyul/AMA/freebase_dump/freebase-rdf-latest.gz"
    reader = FreebaseDumpReaderC()
    reader.open(file)
    Parser = FreebaseDumpParserC()


    for cnt,lvCol in enumerate(reader):

        if 0 == (cnt % 1000):
            print 'read [%d] obj' %(cnt)

        for vCol in lvCol:
            e1 = vCol[0]
            e1 = Parser.DiscardPrefix(e1).decode('utf-8')
            r = Parser.DiscardPrefix(vCol[1]).decode('utf-8')
            e2 = Parser.DiscardPrefix(vCol[2]).decode('utf-8')
            if e1.startswith("m."):
                at_index = e2.find("@")
                url_index = e2.find("^")

                if (at_index != -1 and url_index != -1):
                    e2 = e2[:min(at_index, url_index)]
                elif (at_index != -1):
                    e2 = e2[:at_index]
                elif (url_index != -1):
                    e2 = e2[:url_index]

                print e2


extract_wikiurl()



