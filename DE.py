import math
import random
import numpy as np
import matplotlib.pyplot as plt
class De:
    def __init__(self,F,CR):
        self.F=F
        self.CR=CR
 
 
    def fitness(self,X):
        if self.pdhs(X)>=2:
            y = math.cos(X[:, 1]) + math.sin(X[:, 0]) + X[:, 1] ** 2
        else:
            y=math.cos(X[1]) + math.sin(X[0]) + X[1] ** 2
        return y
    def mutate(self,X):
        P=[]
        n,m=np.shape(X)
        X=np.array(X)
        a,b,c = random.sample(range(0, n), 3)
        #随机选择三个种群进行变异操作
        for i in range(n):
            t=X[a,:]+self.F*(X[b,:]-X[c,:])
            P.append(t)
        P=np.array(P)
        # print(P)
        return P
    def cross(self,X,P):
        NewP=[]
        n, m = np.shape(X)
        for i in range(n):
            temp = []
            for j in range(m):
                # 随机生成一个0-1之间的小数
                c = random.uniform(0, 1)
                # 如果c<=cr,则选择H中个体染色体
                if (c <= self.CR):
                    temp.append(X[i,j])
                else:
                    temp.append(P[i,j])
            # 将新产生的个个体加入新种群
            NewP.append(temp)
        return NewP
 
    def selection(self,X,NewP):
        Newpopulation = []
        NewP=np.array(NewP)
        n, m = np.shape(X)
        for i in range (n):
            if self.fitness(X[i,:])>self.fitness(NewP[i,:]):
                t=NewP[i,:].copy()
            else:
                t=X[i,:].copy()
            # print(t)
            Newpopulation.append(t)
        Newpopulation=np.array(Newpopulation)
        return Newpopulation
    def bestp(self,Newpopulation):
        n, m = np.shape(Newpopulation)
        bestpopulation=[]
        bestfitness=10000
        for i in range(n):
            if self.fitness(Newpopulation[i, :]) < bestfitness:
                bestpopulation=Newpopulation[i,:].copy()
        return bestpopulation,self.fitness(bestpopulation)
 
    def pdhs(self, X):
        m = np.shape(X)
        # 数组维度至少为2D时返回行数，否则返回错误信息
        if X.ndim >= 2:
            return m[0]
        else:
            return 1  # "输入不是二维数组，没有行的概念。"
 
 
#种群数量
N=1000
#特征数
M=2
De=De(F=0.8,CR=0.4)
#种群的生成
population = []
for i in range(N):
    chromosome = []  # 每个个体的染色体
    for j in range(M):
        chromosome.append(random.uniform(-2, 2))  # 自变量的范围是[-4,4]
    population.append(chromosome)
#迭代次数
population=np.array(population)
iters=100
#记录每次迭代的最优种群
besthistory=[]
iterss=[]
#种群更新全过程
for i in range(iters):
    #变异
    P=De.mutate(population)
    #交叉
    NewP=De.cross(population,P)
    #选择产生的下一代种群
    population=De.selection(population,NewP)
    bestpopulation,bestfitness=De.bestp(population)
    besthistory.append(bestfitness)
    iterss.append(i)
    print("第{}次迭代,最优种群为：{},最优适应度：{}".format(i+1,bestpopulation,bestfitness))
#迭代图像绘制
plt.rcParams["font.sans-serif"] = "SimHei"#汉字乱码的解决方法
plt.plot(iterss,besthistory,color='g')
#坐标题目绘制
plt.xlabel("迭代次数")
plt.ylabel("每次迭代的最优适应度")
plt.show()
 
 