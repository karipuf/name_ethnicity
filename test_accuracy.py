from argparse import ArgumentParser

ap=ArgumentParser()
ap.add_argument('-m',help='Model file path')
ap.add_argument('-i',help='Input pickle file to test on')
parsed=ap.parse_args()

import pickle
import pylab as pl
from classify_name import *

# Parsing input parameters
if parsed.m==None:
    # The best so far...
    savePath='good_models/model_800units_0.005lr_20maxNameLen_withsociology'
else:
    savePath=parsed.m

if parsed.i==None:
    testFile='testSets/FilmEth.pkl'
else:
    testFile=parsed.i

LoadNetwork(savePath=savePath)
testSet=pickle.load(open(testFile,'rb'))
#testSet=pickle.load(open("PsychologyEth.pkl",'rb'))

vec=[]
for item in testSet.items():
    vec.append((item[0],','.join([tmp['best'] for tmp in item[1]])))

# The number of items to test (since it can take a while)
nTest=-1

ypred=ClassifyNames([tmp[0] for tmp in vec[:nTest]])
ylab=np.array([tmp[1] for tmp in vec[:nTest]])
acc=len(pl.find(ypred==ylab))/len(vec[:nTest])

print("Accuracy is "+str(100*acc)+"%")

