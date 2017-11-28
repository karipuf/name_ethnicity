import tensorflow as tf
import pickle,re,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Utility functions
def ToInt(instr,maxLen=15):
    basevec=[ord(tmp.lower())-96 for tmp in  re.sub('\s','`',instr)]
    if len(instr)>=maxLen:
        return basevec[:maxLen]
    else:
        return basevec+[0]*(maxLen-len(basevec))
        

# Initializations
nUnits=600
lr=.005

savePath='currentModel_'+str(nUnits)+'units_'+str(lr)+'lr'
#savePath='currentModel_300units_100hidden_0.005lr'
savePath='bestSoFar'

# Extracting features
neths=['Asian,GreaterEastAsian,EastAsian','Asian,GreaterEastAsian,Japanese','Asian,IndianSubContinent','GreaterAfrican,Africans','GreaterAfrican,Muslim','GreaterEuropean,British','GreaterEuropean,EastEuropean','GreaterEuropean,Jewish','GreaterEuropean,WestEuropean,French','GreaterEuropean,WestEuropean,Germanic','GreaterEuropean,WestEuropean,Hispanic','GreaterEuropean,WestEuropean,Italian','GreaterEuropean,WestEuropean,Nordic']

#fnames=['EcologyEth.pkl','PolSciEth.pkl','OceanEth.pkl','imdbeths.pkl','AccountingEth.pkl','LanguageEth.pkl']

#names=[]
#eths=[]
#for fname in fnames:
#    d=pickle.load(open(fname,"rb"))
#    for tmp in d.items():
#       eths.append(','.join([tmp2['best'] for tmp2 in tmp[1]]))

le=LabelEncoder().fit(neths)

try:
    sess.close()
except:
    pass

try:
    tf.reset_default_graph()
except:
    pass

sess=tf.Session()
loader=tf.train.import_meta_graph(savePath+'.meta')
loader.restore(sess,savePath)
opt=sess.graph.get_operation_by_name('Adam')
loss=sess.graph.get_tensor_by_name('Mean:0')
x=sess.graph.get_tensor_by_name('Placeholder:0')
y=sess.graph.get_tensor_by_name('Placeholder_1:0')
yhat=sess.graph.get_tensor_by_name('Softmax:0')

#sess.run(tf.global_variables_initializer())
def ClassifyName(inname):
    return le.inverse_transform(np.argmax(sess.run(yhat,feed_dict={x:[ToInt(inname)]})))
