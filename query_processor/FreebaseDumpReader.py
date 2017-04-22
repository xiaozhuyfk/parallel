'''
Created on Nov 4, 2015 10:43:49 AM
@author: cx

what I do:
    I read freebase dump in gzip format
what's my input:
    Fb dump
what's my output:
    each time all triples for a document

'''






import gzip


class KeyBasedFileReaderC(object):
    def Init(self):
        self.InFile = ""
        self.UseGzip = False
        self.lKeyIndex = [0]
        self.Spliter = '\t'
        self.MaxLinePerKey = 1000000
        self.LastvCol = []
        self.InName = ""
    def __init__(self):
        self.Init()
        
        
    def open(self,InName,mode = 'r'):
        self.InName = InName
        if self.UseGzip:
            self.InFile = gzip.open(InName,mode)
        else:
            self.InFile = open(InName,mode)
            
    def empty(self):
        return "" == self.InName
            
    def close(self):
        if not self.empty():
            self.InFile.close()
            self.InName = ""
    
    
    def GenerateKey(self,vCol):
        key = ""
        for i in self.lKeyIndex:
            key += vCol[i] + self.Spliter
        return key.strip(self.Spliter)
        
        
    def ReadNextKey(self):
        lvCol = [] #one object's all triples
        CurrentKey = ""
        if self.LastvCol != []:
            lvCol.append(self.LastvCol)
            CurrentKey = self.GenerateKey(self.LastvCol)
            self.LastvCol = []
        cnt = 0
        for line in self.InFile:
            vCol = line.strip().split(self.Spliter)
            if [] == vCol:
                continue
            ThisKey = self.GenerateKey(vCol)
#             print "read [%s]" %(line.encode('utf-8','ignore'))

            if CurrentKey == "":
                CurrentKey = ThisKey
            if ThisKey != CurrentKey:
                self.LastvCol = vCol
                break
            if cnt < self.MaxLinePerKey:
                lvCol.append(vCol)
            cnt += 1
        return lvCol
    
    
    def __iter__(self):
        return self
    
    def next(self):
        lvCol = self.ReadNextKey()
        if lvCol == []:
            raise StopIteration
        return lvCol    
            
        
 
class FreebaseDumpReaderC(KeyBasedFileReaderC):
    
    
    def Init(self):
        super(FreebaseDumpReaderC,self).Init()
        self.UseGzip = True #always
    
            
    def ReadNextKey(self):
        lvCol = super(FreebaseDumpReaderC,self).ReadNextKey()
        lvCol = [self.ProcessOneLine(vCol) for vCol in lvCol]
        return lvCol
        
    def ProcessOneLine(self,vCol):
        if len(vCol) < 3:
            return []
#         print "processing %s" %(json.dumps(vCol))
#        vCol = [DiscardPrefix(col) for col in vCol[:3]]      
        return vCol