import tensorflow as tf
import pickle,re
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
nUnits=100
nEthnicities=13
lr=.001
nIters=30000
nDisp=10
#savePath='currentModel'
savePath='currentModelBidirectional'
existingModel=False
minibatch=128
valProp=.1

# Extracting features
fnames=['EcologyEth.pkl','PolSciEth.pkl','OceanEth.pkl','imdbeths.pkl','AccountingEth.pkl']

names=[]
eths=[]
for fname in fnames:
    d=pickle.load(open(fname,"rb"))
    for tmp in d.items():
       names.append(tmp[0].lower())
       eths.append(','.join([tmp2['best'] for tmp2 in tmp[1]]))


xnum=np.array([ToInt(tmp) for tmp in names])
le=LabelEncoder().fit(eths)
ynum=np.array(le.transform(eths))

# Creating the network
x=tf.placeholder(tf.int32,shape=(None,None))
xoh=tf.one_hot(x,28,dtype=tf.float32)
y=tf.placeholder(tf.int32,shape=(None,))
yoh=tf.one_hot(y,nEthnicities,dtype=tf.float32)

cell=tf.contrib.rnn.BasicLSTMCell(nUnits)
cell_bw=tf.contrib.rnn.BasicLSTMCell(nUnits)

rnn=tf.nn.dynamic_rnn(cell,xoh,dtype=tf.float32)
#rnn=tf.nn.bidirectional_dynamic_rnn(cell,cell_bw,xoh,dtype=tf.float32)

wout=tf.Variable(tf.truncated_normal((nUnits,nEthnicities),stddev=.001))
#wout=tf.Variable(tf.truncated_normal((nUnits*2,nEthnicities),stddev=.001))
bout=tf.Variable(tf.zeros((nEthnicities)))

logitsTraining=tf.add(tf.nn.dropout(tf.matmul(rnn[1].h,wout),0.5),bout)
logits=tf.add(tf.matmul(rnn[1].h,wout),bout)
#logits=tf.add(tf.matmul(tf.concat([tmp.h for tmp in rnn[1]],axis=1),wout),bout)
yhat=tf.nn.softmax(logits)

lossTraining=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logitsTraining,labels=yoh))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=yoh))
opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(lossTraining)
opt2=tf.train.AdamOptimizer(learning_rate=lr/2).minimize(lossTraining)

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

for count in range(nIters):

    randInds=np.random.choice(range(len(ynumTrain)),minibatch,replace=False)
    if count<(nIters/2):
        sess.run(opt,feed_dict={x:xnumTrain[randInds],y:ynumTrain[randInds]})
    else:
        sess.run(opt2,feed_dict={x:xnumTrain[randInds],y:ynumTrain[randInds]})
    if count % nDisp==0:
        print("Iteration# "+str(count)+", validation error is: "+str(sess.run(loss,feed_dict={x:xnumVal,y:ynumVal})))

saver=tf.train.Saver()
saver.save(sess,savePath);

#sess.close()
