import tensorflow as tf
import pickle,re,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Utility functions
def ToInt(instr,maxLen=20):
    basevec=[ord(tmp.lower())-96 for tmp in  re.sub('\s','`',instr)]
    if len(instr)>=maxLen:
        return basevec[:maxLen]
    else:
        return basevec+[0]*(maxLen-len(basevec))
        

# Initializations
nameLen=20
savePath='currentBest_model_800units_0.005lr_20maxNameLen'
tensors={}

# Creating label encoder for the ethnicities
neths=['Asian,GreaterEastAsian,EastAsian','Asian,GreaterEastAsian,Japanese','Asian,IndianSubContinent','GreaterAfrican,Africans','GreaterAfrican,Muslim','GreaterEuropean,British','GreaterEuropean,EastEuropean','GreaterEuropean,Jewish','GreaterEuropean,WestEuropean,French','GreaterEuropean,WestEuropean,Germanic','GreaterEuropean,WestEuropean,Hispanic','GreaterEuropean,WestEuropean,Italian','GreaterEuropean,WestEuropean,Nordic']

le=LabelEncoder().fit(neths)


def LoadNetwork(savePath=savePath,sess=None,resetGraph=True):

    if resetGraph:
        try:
            tf.reset_default_graph()
        except:
            pass

    if sess==None:
        try:
            sess.close()
        except:
            pass
        sess=tf.Session()

    # Loading in the network
    loader=tf.train.import_meta_graph(savePath+'.meta')
    loader.restore(sess,savePath)
    
    tensors['opt']=sess.graph.get_operation_by_name('Adam')
    tensors['loss']=sess.graph.get_tensor_by_name('Mean:0')
    tensors['x']=sess.graph.get_tensor_by_name('Placeholder:0')
    tensors['y']=sess.graph.get_tensor_by_name('Placeholder_1:0')
    tensors['yhat']=sess.graph.get_tensor_by_name('Softmax:0')
    tensors['sess']=sess
    
    return None

#sess.run(tf.global_variables_initializer())
def ClassifyName(inname):

    sess=tensors['sess']
    yhat=tensors['yhat']
    x=tensors['x']
    
    return le.inverse_transform(np.argmax(sess.run(yhat,feed_dict={x:[ToInt(inname,maxLen=nameLen)]})))

def ClassifyNames(innames):

    sess=tensors['sess']
    yhat=tensors['yhat']
    x=tensors['x']
        
    namevecs=[ToInt(tmp,maxLen=nameLen) for tmp in innames]
    return le.inverse_transform(np.argmax(sess.run(yhat,feed_dict={x:namevecs}),axis=1))

LoadNetwork(savePath='model_800units_0.005lr_20maxNameLen_withsociology');
