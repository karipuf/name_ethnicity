# Script to train name ethnicity classifier

from argparse import ArgumentParser
ap=ArgumentParser()
ap.add_argument('-n',help='Number of hidden units (default 800)')
ap.add_argument('-lr',help='Learning rate (default 0.005)')
ap.add_argument('-d',help='Display output every n iterations (note: for now this is also the interval at which models are evaluated and saved to "currentBest"')
ap.add_argument('-m',help='Set minibatch size (default 128)')
ap.add_argument('-i',help='Number of training iterations (default 30000)')
ap.add_argument('-mnl',help='Maximum length of the name vector (default 20)')
parsed=ap.parse_args()

import glob
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
if parsed.n==None:
    nUnits=800
else:
    nUnits=int(parsed.n)

if parsed.lr==None:
    lr=.005
else:
    lr=float(parsed.lr)

if parsed.d==None:
    nDisp=250
else:
    nDisp=int(parsed.d)

if parsed.m==None:
    minibatch=128
else:
    minibatch=int(parsed.m)

if parsed.i==None:
    nIters=30000
else:
    nIters=int(parsed.i)

if parsed.mnl==None:
    nameLen=20
else:
    nameLen=int(parsed.mnl)


nEthnicities=13
fnames=glob.glob('trainingSets/*pkl')
savePath='model_'+str(nUnits)+'units_'+str(lr)+'lr_'+str(nameLen)+'maxNameLen_'+str(len(fnames))+'trainingfiles_'+str(minibatch)+'minibatch_'+str(nIters)+'iterations'
existingModel=False
valProp=.1

names=[]
eths=[]
for fname in fnames:
    d=pickle.load(open(fname,"rb"))
    for tmp in d.items():
       names.append(tmp[0].lower())
       eths.append(','.join([tmp2['best'] for tmp2 in tmp[1]]))


xnum=np.array([ToInt(tmp,maxLen=nameLen) for tmp in names])
le=LabelEncoder().fit(eths)
ynum=np.array(le.transform(eths))

# Creating the network
x=tf.placeholder(tf.int32,shape=(None,None))
xoh=tf.one_hot(x,28,dtype=tf.float32)
y=tf.placeholder(tf.int32,shape=(None,))
yoh=tf.one_hot(y,nEthnicities,dtype=tf.float32)

cell=tf.contrib.rnn.BasicLSTMCell(nUnits)
#cell_bw=tf.contrib.rnn.BasicLSTMCell(nUnits)

rnn=tf.nn.dynamic_rnn(cell,xoh,dtype=tf.float32)
#rnn=tf.nn.bidirectional_dynamic_rnn(cell,cell_bw,xoh,dtype=tf.float32)

wout=tf.Variable(tf.truncated_normal((nUnits,nEthnicities),stddev=.001))
bout=tf.Variable(tf.zeros((nEthnicities)))

#wout=tf.Variable(tf.truncated_normal((nUnits*2,nEthnicities),stddev=.001))


logitsTraining=tf.add(tf.nn.dropout(tf.matmul(rnn[1].h,wout),0.6),bout)
logits=tf.add(tf.matmul(rnn[1].h,wout),bout)
#logits=tf.add(tf.matmul(tf.concat([tmp.h for tmp in rnn[1]],axis=1),wout),bout)
yhat=tf.nn.softmax(logits)

lossTraining=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logitsTraining,labels=yoh))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=yoh))
opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(lossTraining)
opt2=tf.train.AdamOptimizer(learning_rate=lr/2).minimize(lossTraining)
opt3=tf.train.AdamOptimizer(learning_rate=lr/4).minimize(lossTraining)

try:
    sess.close()
except:
    pass
sess=tf.Session()

if existingModel:
    tf.reset_default_graph()
    loader=tf.train.import_meta_graph(savePath+'.meta')
    loader.restore(sess,savePath)
    opt=sess.graph.get_operation_by_name('Adam')
    loss=sess.graph.get_tensor_by_name('Mean:0')
    x=sess.graph.get_tensor_by_name('Placeholder:0')
    y=sess.graph.get_tensor_by_name('Placeholder_1:0')

sess.run(tf.global_variables_initializer())

# Setting aside validation set
valInds=np.random.rand(len(ynum))<valProp
trainInds=np.logical_not(valInds)

xnumTrain=xnum[trainInds];ynumTrain=ynum[trainInds]
xnumVal=xnum[valInds];ynumVal=ynum[valInds]

curMin=0.9
for count in range(nIters):

    randInds=np.random.choice(range(len(ynumTrain)),minibatch,replace=False)
    if count<(nIters/2):
        sess.run(opt,feed_dict={x:xnumTrain[randInds],y:ynumTrain[randInds]})
    else:
        if count<(nIters*0.75):
            sess.run(opt2,feed_dict={x:xnumTrain[randInds],y:ynumTrain[randInds]})
        else:
            sess.run(opt3,feed_dict={x:xnumTrain[randInds],y:ynumTrain[randInds]})
    if count % nDisp==0:
        err=sess.run(loss,feed_dict={x:xnumVal,y:ynumVal})
        print(savePath+", Iteration# "+str(count)+", validation error is: "+str(err))
        if err<curMin:
            print("New low error! Caching the model")

            curMin=err
            saver=tf.train.Saver()
            saver.save(sess,"currentBest_"+savePath);
        sys.stdout.flush()

        

saver=tf.train.Saver()
saver.save(sess,savePath);

#sess.close()
