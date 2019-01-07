# -*- coding: utf-8 -*-
#from pymongo import MongoClient
from gensim.models import word2vec
import random
import os
import jieba
import re
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

def getDataFromMongoDB(RawDataFileName):
    if os.path.exists(RawDataFileName):
        print("评论数据已经从MongoDB导出到本地")
        print("可以加载爬虫经MongoDB导出到本地评论数据文件：%s" % (RawDataFileName))
        commentsData=[]
        with open(RawDataFileName, "r", encoding="utf-8") as f:
            #for line in f.read():
            try:
                lines = f.read()
                #print(lines)
                #print(type(lines))
                items=re.findall("\n-----------\n(.*?)\n-----------\n",lines)
                #print(items)
                for item in items:
                    content=item.split("###")[0]
                    score = int(item.split("###")[1])
                    #print(content)
                    #print(score)
                    commentsData.append((content,score))
            except:
                pass
                #print(line.strip())
        return commentsData


    else:
        pass
        #主要功能是连接MongoDB数据库，并从中获取相关数据
        #连接MongoDB
#         conn=MongoClient('127.0.0.1',27017)
#         db=conn.JDSpider_Comment
#         comments=db.comment_detail

#         # 去重，并且通过三目运算符，将3-5分的评分定为积极(用1表示)，1-2分定为消极(用0表示)
#         # 优化时候，可以考虑多分类问题
#         commentsData = list(set([(-1,-1) if ("此用户未填写评价内容" in comment["content"])
#                                 else (comment["content"],1 if(comment["score"]>=3) else 0)
#                                 for comment in comments.find()]))
#         with open(RawDataFileName,"w+" ,encoding="utf-8") as f:
#             for item in commentsData:
#                 f.write("\n-----------\n"+str(item[0])+"###"+str(item[1]))#不用用\n，因为一段评论中也可能有很多的\n

        return commentsData

def dataBalance(data):
    #进行正负样本平衡，此处采用向下采样,也就是取少的

    positive=[]
    negative=[]
    for eveItem in data:
        if eveItem[1] ==1:
            positive.append(eveItem)
        elif eveItem[1] ==0:
            negative.append(eveItem)
    #打乱顺序，防止采集的时候获得到的数据，相邻的是类似产品的
    random.shuffle(positive)
    random.shuffle(negative)

    positiveNum=len(positive)
    negativeNum=len(negative)

    if positiveNum>negativeNum:
        positive = positive[0:len(negative)]
    else:
        negative = negative[0:len(positive)]
    return (positive, negative, (positiveNum, negativeNum))

def word2vecFun(textName,modelName):
    if os.path.exists(modelName):
        print("word2vec模型文件已经存在")
        print("可以加载本地word2vec模型文件：%s"%(modelName))
        model=word2vec.Word2Vec.load(modelName)
    else:
        print("word2vec模型文件不存在")
        print("需要训练本地word2vec模型文件：%s" % (modelName))
        model = word2vec.Word2Vec(word2vec.LineSentence(textName),
                                  min_count=10, window=10)
        model.save(modelName)
    return model

def getStopWords(stopWordsName):
    with open(stopWordsName,encoding="utf-8") as f:
        stopWords=[word.replace("\n","") for word in f.readlines()]
    return stopWords

def cutWords(data,stopWords):
    #结巴分词,只分汉字
    dataCut=[]
    for eveSentence in data:
        cutWord= jieba.cut("".join(re.findall(r'[\u4e00-\u9fa5]', eveSentence[0])))
        tempWord=[]
        for eveWord in cutWord:
            if eveWord not in stopWords:
                tempWord.append(eveWord)
        dataCut.append((tempWord, eveSentence[1]))
    return dataCut

def getWordLen(data,percent):
    #分词后的词长度,输入时cutWords的dataCut
    # 在percent分位数上的长度是多少？，相当于截断，count这么多个分好的词
    tempList = [len(eveItem[0]) for eveItem in data]
    listLen = len(tempList)
    tempCount = int(listLen * percent)#一共多少个分好的词的乘上一个比例值
    tempList.sort()#正序的长度
    drawHist(tempList, 'len', 'count')  # 直方图展示
    return tempList[tempCount]

def drawHist(myList,Xlabel,Ylabel):
    # 参数依次为list,title,X轴标签,Y轴标签,XY轴的范围
    plt.hist(myList)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.show()

def maxList(listData):
    #暂时没用了
    temp = 0
    for i in listData:
        if listData.count(i) > temp:
            maxData = i
            temp = listData.count(i)
    return maxData

def word2Index(TEXTNAME):
    #暂时没用，用的是后面的getIndexVectory函数
    with open(TEXTNAME) as f:
        tempData=[eveSentence.strip() for eveSentence in f.readlines()]
    textList = []
    for eveSentence in tempData:
        for eveWord in eveSentence.split(" "):
            textList.append(eveWord)
    textList = list(set(textList))
    return textList

def getSentenceVec(sentence,count,textIndex):
    #输入是cutWords分词后每条分好词的句子
    evesentenceVec= np.zeros((count),dtype='int32')#根据前面的截断，认为每个句子都最长只有count这么多个分好的词
    i=0
    # 这里由于取了count值，在这里是直接进行了切割，这样会导致准确度下降
    # 优化的时候可以考虑此处用tf-idf来看一下每个词的"影响度"，根据影响度（权重），进行排序，取前count个
    # 也就是不仅仅按照句子的顺序，从前到后取countcount这么多个分好的词
    for eve in sentence[0:count]:
        try:
            evesentenceVec[i] = textIndex.index(eve)
        except:
            evesentenceVec[i] = -1#有些句子中分词后的词个数本来就少于count的句子，后面都置位-1
        i=i+1
    return evesentenceVec

def getIndexVectory(model):
    indexData = []
    vectoryData = []
    for eve in model.wv.vocab:
        indexData.append(eve)
        vectoryData.append(list(model[eve]))

    return (indexData, np.array(vectoryData))#tf查表需要np.array格式的支持

#--------------------------------------------
#数据处理
TEXTNAME = "jd_comments_181007_cutted.txt"
MODELNAME = "jd_comments_181_7_model.model"
STOPWORDS = "StopwordsCN.txt"
RawDataFileName="RawData_comments.txt"
print("STEP 模型加载")
model = word2vecFun(TEXTNAME, MODELNAME)
print("STEP 索引与词向量生成")
indexData, vectoryData = getIndexVectory(model)
print("STEP 停用词加载")
stopWords = getStopWords(STOPWORDS)
print("STEP 数据加载")
data = getDataFromMongoDB(RawDataFileName)
print(data[0])
print(data[-1])
print("total data count:",len(data))

print(list(model["京东"]))
print(len(list(model["京东"])))
print(list(model["连衣裙"]))
print(len(list(model["连衣裙"])))


print("STEP 下采样数据平衡")
positive,negative,dataCount = dataBalance(data)
print(dataCount)
print("STEP 分词操作")
positiveCut = cutWords(positive,stopWords)
negativeCut = cutWords(negative,stopWords)

print("STEP 获得句长分布")
userData = positiveCut + negativeCut
count = getWordLen(userData,0.80)

print("STEP 数据处理")
sentenceVec = []
for eveSentence in positiveCut:
    sentenceVec.append((getSentenceVec(eveSentence[0],count,indexData),eveSentence[1]))
for eveSentence in negativeCut:
    sentenceVec.append((getSentenceVec(eveSentence[0],count,indexData),eveSentence[1]))

print(sentenceVec[:5])
#-----------------------------------------
# 开始建模训练

print("STEP 正在准备建模参数")
BATCHSIZE =50
LSTMUNITS=64
NUMCLASSES=2
ITERATIONS=10000#50000
MAXSEQLENGTH=count
NUMDIMENSIONS=300#gensim默认100维啊
print("STEP 正在建模")

#batch 方法
def getTrainBatch(sentenceVec):
    labels=[]
    arr=np.zeros([BATCHSIZE,MAXSEQLENGTH])
    for i in range(BATCHSIZE):
        tempData=random.choice(sentenceVec)
        if tempData[1]==0:
            labels.append([1, 0])##分类任务的y写法，预测则不是这样
        else:
            labels.append([0, 1])
        arr[i]=np.array(tempData[0])

    return arr,labels

# 折线图
def drawLine(list1,list2,title1,title2):

    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(list1)
    ax.set_title(title1)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(list2)
    ax.set_title(title2)
    plt.show()

#训练的graph,网络图
tf.reset_default_graph()
labels=tf.placeholder(tf.float32,[BATCHSIZE,NUMCLASSES])
inputData=tf.placeholder(tf.int32,[BATCHSIZE,MAXSEQLENGTH])#序号值

data = tf.Variable(tf.zeros([BATCHSIZE, MAXSEQLENGTH, NUMDIMENSIONS]),dtype=tf.float32)
data = tf.nn.embedding_lookup(vectoryData,inputData+1)

lstmCell=tf.contrib.rnn.BasicLSTMCell(LSTMUNITS)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value,_=tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)
value=tf.transpose(value,[1,0,2])#也就是从[0,1,2]变为[1,0,2]

#取最终的结果值
last=tf.gather(value,int(value.get_shape()[0])-1)
weight = tf.Variable(tf.truncated_normal([LSTMUNITS, NUMCLASSES]))
bias = tf.Variable(tf.constant(0.1, shape=[NUMCLASSES]))
prediction=(tf.matmul(last,weight)+bias)

correctPred=tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
accuracy=tf.reduce_mean(tf.cast(correctPred,tf.float32))

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=labels))
optimizer=tf.train.AdamOptimizer().minimize(loss)

sess=tf.InteractiveSession()
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

#-----------------------------------------------
print("STEP 开始训练")
lossList = []
accuracyList = []

for i in range(ITERATIONS):
    nextBatch,nextBatchLabels= getTrainBatch(sentenceVec)
    sess.run(optimizer,{inputData:nextBatch,labels:nextBatchLabels})

    loss_=sess.run(loss,{inputData:nextBatch,labels:nextBatchLabels})
    accuracy_=sess.run(accuracy,{inputData:nextBatch,labels:nextBatchLabels})

    lossList.append(loss_)
    accuracyList.append(accuracy_)

    if(i%500==0 and i!=0):
        print("iteration {}/{}...".format(i+1,ITERATIONS),
              "loss {}...".format(loss_),
              "accuracy {}...".format(accuracy_))

    if (i % 500 == 0 and i != 0):
        save_path=saver.save(sess,"models/%s.lstm_model"%(MODELNAME),
                             global_step=i)
        print("saved to %s"%save_path)

drawLine(lossList,accuracyList,"LOSS LINE", "ACCURACY LINE")


