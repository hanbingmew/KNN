
# coding: utf-8

# In[ ]:


import os
import math
from functools import reduce
from heapq import *
from time import clock
def file2matrix(dirname):#dirname是文件夹名称，即trainingDigits或testDigits
    datamatrix=[]
    label=[]
    for root, dirs, files in os.walk('/home/zijun/digits/%s'% dirname):#这个路径是存放数据的路径，可以根据具体位置更改
        for filename in files:
            data_label=int(filename[0])
            str_line=''
            int_line=[]
            with open('/home/zijun/digits/%s/%s' % (dirname,filename),'r') as fr:
                data=fr.readlines()
            for i in range(len(data)):
                data[i]=data[i].rstrip('\n')
                str_line+=data[i]
            for i in range(len(str_line)):
                int_line.append(int(str_line[i]))  
            datamatrix.append(int_line)
            label.append(data_label)
    return datamatrix,label

def getEuclidDist(x1,x2):
    dist=0
    for i in range(len(x1)):
        dist+=(x1[i]-x2[i])**2
    return math.sqrt(dist)

def getManDist(x1,x2):
    dist=0
    for i in range(len(x1)):
        dist+=abs(x1[i]-x2[i])
    return dist

def getKNeighbors(k,datamatrix,label,new_data):
    dist=[]
    for i in range(len(datamatrix)):
        dist.append((getEuclidDist(datamatrix[i],new_data),i,label[i]))#如果用曼哈顿距离，把getEuclidDist换成getManDist
    neighbors=sorted(dist)[:k]
    return neighbors

def makePredict(neighbors):
    predict_dict={}
    for i in neighbors:
        if i[2] not in predict_dict:
            predict_dict[i[2]]=1
        else:
            predict_dict[i[2]]+=1
    return sorted(predict_dict.items(),key=lambda x:x[1],reverse=True)[0][0]

def makePredict_weighted(neighbors):
    predict_dict={}
    for i in range(len(neighbors)):
        if i==0:
            weight=1
        else:
            try:
                weight=(neighbors[-1][0]-neighbors[i][0])/(neighbors[-1][0]-neighbors[0][0])
            except ZeroDivisionError:
                weight=1
        if neighbors[i][2] not in predict_dict:
            predict_dict[neighbors[i][2]]=weight
        else:
            predict_dict[neighbors[i][2]]+=weight
    return sorted(predict_dict.items(),key=lambda x:x[1],reverse=True)[0][0]

def calAccuracy(predict_vec,label):
    right_num=0
    for i in range(len(predict_vec)):
        if predict_vec[i]==label[i]:
            right_num+=1
    return right_num/len(predict_vec)

class kdNode:
    def __init__(self,data,left,right,split):
        self.data=data
        self.left=left
        self.right=right
        self.split=split
    
class kdTree:
    def __init__(self,data_set):#data_set的格式是[(i,data[i]) for i in range(len(data))]
        def createNode(data_set):
            if len(data_set)==0:#数据集长度为0时终止
                return None
            else:
                dim=[]
                for i in range(len(data_set[0][1])):#求出数据矩阵的所有列向量
                    dim.append([data_set[j][1][i] for j in range(len(data_set))])
                variance=list(map(getVariance,dim))#求出所有列的方差
                split=variance.index(max(variance))#选择方差最大的维度为切分维度   
                data_set.sort(key=lambda x:x[1][split])#根据切分维度对数据排序
                split_data=data_set[len(data_set)//2]
                #把排序后数据集分成两半，分别用来建立当前结点的左右孩子，如此递归地建立kd树
                root=kdNode(split_data,createNode(data_set[:len(data_set)//2]),createNode(data_set[len(data_set)//2+1:]),split)
                return root
                
        def getVariance(l):
            sum_1=reduce((lambda x,y:x+y),l)
            mean=sum_1/len(l)
            sum_2=reduce((lambda x,y:x+y),list(map((lambda x:x**2),l)))
            return (sum_2/len(l)-mean**2)
        
        self.root=createNode(data_set)
        
def search_knn(tree,point,k):#递归搜索
    result=[]
    def search_node(node,point,result,k):
        if not node:#结点为空时终止
            return
        else:
            node_dist=getEuclidDist(node.data[1],point)#求当前结点保存的数据与目的数据的距离
            item=(-node_dist,node.data)#把距离的相反数和数据打包成元组
            #当优先队列元素已经达到k时，只有当前结点的距离小于优先队列中的最大距离时才把当前结点加入优先队列，替换队列中距离最大的结点
            if len(result)>=k:
                if -node_dist>result[0][0]:
                    heapreplace(result, item)
            else:
                heappush(result, item)#队列元素数量小于k时直接把当前结点加入
            #递归搜索目的数据所在的孩子结点，更新优先队列
            if node.data[1][node.split]>=point[node.split]:
                search_node(node.left,point,result,k)
                next_node=node.right
            else:
                search_node(node.right,point,result,k)
                next_node=node.left
            #如果优先队列中的最大距离小于目的结点到分割超平面的距离，就不需要搜索另一个孩子结点，否则搜索另一个孩子节点
            if -abs(node.data[1][node.split]-point[node.split])>result[0][0] or len(result)<k:
                search_node(next_node,point,result,k)
    
    search_node(tree.root,point,result,k)
    return result    

def search_knn_norecur(tree,point,k):#非递归搜索
    result=[]
    def search_node(node,point,result,k):
        node_stack=[]
        temp_node=node
        #从根结点开始找到目的数据所在的叶子结点，中间所有路过的结点都压入node_stack
        while temp_node:
            node_stack.append(temp_node)
            if temp_node.data[1][temp_node.split]>=point[temp_node.split]:
                temp_node=temp_node.left
            else:
                temp_node=temp_node.right
        #分别处理node_stack中的每个结点，对于每个结点求距离并更新优先队列
        while len(node_stack)>0:
            node=node_stack.pop()
            node_dist=getEuclidDist(node.data[1],point)
            item=(-node_dist,node.data)
            if len(result)>=k:
                if -node_dist>result[0][0]:
                    heapreplace(result, item)
            else:
                heappush(result, item)
            #如果优先队列中的最大距离大于目的结点到分割超平面的距离，把未搜索的另一个子结点压栈
            if -abs(node.data[1][node.split]-point[node.split])>result[0][0] or len(result)<k:
                if node.data[1][node.split]>=point[node.split]:
                    if node.right:
                        node=node.right
                    else:
                        node=None
                else:
                    if node.left:
                        node=node.left
                    else:
                        node=None
            #如果压栈的另一个子结点不是叶子结点，则递归地查找到目的数据所在的叶子结点，路过的所有节点均压栈
            if node:
                while node.left or node.right:
                    node_stack.append(node)
                    if node.data[1][node.split]>=point[node.split]:
                        if not node.left:
                            break
                        else:
                            node=node.left
                    else:
                        if not node.right:
                            break
                        else:
                            node=node.right
                if node.left==None and node.right==None:
                    node_stack.append(node)
    search_node(tree.root,point,result,k)
    return result

if  __name__=='__main__':
    train_data,train_label=file2matrix('trainingDigits')
    test_data,test_label=file2matrix('testDigits')
    predict_vec=[]
    t0=clock()
    for i in range(len(test_data)):
        neighbors=getKNeighbors(3,train_data,train_label,test_data[i])
        predict_vec.append(makePredict(neighbors))#如果要用带权重的的KNN，把makePredict改成makePredict_weighted
    acc=calAccuracy(predict_vec,test_label)
    t1=clock()
    print('accuracy:%s' % acc)
    print('time:%s' % (t1-t0))
    
    
#如果测试kd树，用下面这段代码，注释掉上面的  
'''
if  __name__=='__main__':
    train_data,train_label=file2matrix('trainingDigits')
    test_data,test_label=file2matrix('testDigits')
    dataset=[(i,test_data[i]) for i in range(len(test_data))]
    t0=clock()
    kd = kdTree(dataset)
    predict_vec=[]
    for i in range(len(test_data)):
        result=search_knn(kd,test_data[i],5)  #如果用非递归搜索，把search_knn改成search_knn_norecur
        neighbors=[(-j[0],j[1][0],test_label[j[1][0]]) for j in result]
        predict_vec.append(makePredict(neighbors))
    acc=calAccuracy(predict_vec,test_label)
    t1=clock()
    print('accuracy:%s' % acc)
    print('time:%s' % (t1-t0))
'''

