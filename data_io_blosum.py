#! usr/bin/python
## CNN model for cancer-specific sequence prediction

import sys,os
import numpy as np

def read_file(filename):
   g=open(filename,"r")
   Train_data=[]
   for line in g.readlines():
      Train_data.append(line.strip())
   Train_data=np.array(Train_data)
   return Train_data 

def blosum_matrix(blosum_file):
    blosumfile = open(blosum_file, "r")
    blosum = {}
    B_idx = 99
    Z_idx = 99
    X_idx = 99
    star_idx = 99
    for l in blosumfile:
        l = l.strip()
        if l[0] != '#':
            l= list(filter(None,l.strip().split(" ")))
            if (l[0] == 'A') and (B_idx==99):
                B_idx = l.index('B')
                Z_idx = l.index('Z')
                X_idx = l.index('X')
                star_idx = l.index('*')
            else:
                aa = str(l[0])
                if (aa != 'B') & (aa != 'Z')& (aa != 'X') & (aa != '*'):
                    tmp = l[1:len(l)]
                    tmp2 = []
                    for i in range(0, len(tmp)):
                        if (i != B_idx) & (i != Z_idx) & (i != X_idx) & (i != star_idx):
                            tmp2.append(float(tmp[i]))
                    blosum[aa]=tmp2
    blosumfile.close()
    return(blosum)
    
def enc_list_bl(aa_seqs, Nm, blosum):    
    Seq=list(aa_seqs)
    Ns=len(Seq)
    OHE=np.zeros([20,Nm])
    for ii in range(Ns):
        aa=Seq[ii]
        OHE[0:20,ii]=blosum[aa]
    OHE=OHE.astype(np.float32)
    return OHE
    
def GetFeatureLabels(CancerCDR3s, NonCancerCDR3s, Nm, blosum):
    nt=len(CancerCDR3s)
    nc=len(NonCancerCDR3s)
    FeatureDict={}
    LabelDict={}
    Labels=[1]*nt+[0]*nc
    Labels=np.array(Labels)
    Labels=Labels.astype(np.int32)
    data=[]
    for ss in CancerCDR3s:
            data.append(enc_list_bl(ss,Nm,blosum))
    for ss in NonCancerCDR3s:
            data.append(enc_list_bl(ss,Nm,blosum))
    data=np.array(data)
    FeatureDict=data
    LabelDict=Labels
    return FeatureDict, LabelDict
        