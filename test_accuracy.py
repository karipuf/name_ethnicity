import pickle
import pylab as pl
from classify_name import *

testSet=pickle.load(open("FilmEth.pkl",'rb'))
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

