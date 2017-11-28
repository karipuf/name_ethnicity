import pickle
import pylab as pl
from classify_name import *

testSet=pickle.load(open("FilmEth.pkl",'rb'))
vec=[]
for item in testSet.items():
    vec.append((item[0],','.join([tmp['best'] for tmp in item[1]])))

# The number of items to test (since it can take a while)
nTest=2500

ypred=np.array([ClassifyName(tmp[0]) for tmp in vec[:nTest]])
ylab=np.array([tmp[1] for tmp in vec[:nTest]])
acc=len(pl.find(ypred==ylab))/nTest

print("Accuracy is "+str(100*acc)+"%")

