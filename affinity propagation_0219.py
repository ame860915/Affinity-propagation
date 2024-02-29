from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

# #############################################################################
# Generate sample data
'''
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, labels_true = make_blobs(
    n_samples=1000, centers=centers, cluster_std=0.4, random_state=0
)
print(X)
'''
#'''
# Import data
np.set_printoptions(threshold = 11251)

data = pd.read_csv('0901_Trainingdata_v1.txt',
                   names = ['B', 'G', 'R', 'S', 'target'],
                   sep = '\t+', engine = 'python')

target = data['target'].values
label_names = {0:'Background', 1:'Paper', 2:'Bud', 3:'Root', 4:'Seed', 5:'Dish'}

#print('pre\n', data)
data = data[data.columns[:-1]]
#print('aft\n',data)
data = data / 255
data = data.values
print(data)
#'''
#'''
# #############################################################################
#計算每個點 與 其他所有點之間的歐基里德距離
def euclideanDistance(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    #print(X)
    return np.sqrt(np.sum(X-Y)**2)


# #############################################################################
#計算 Similarity(相似度) : 數據點i 與 數據點j 的相似值，
#                         值越大代表  數據點i 與 數據點j 越相近，
#                         AP演算法中理解為 數據點j 作為 數據點i 的聚類中心的能力
def computeSimilarity(datalist):
    num = len(datalist)
    Similarity = []
    for pointX in datalist:
        dists = []
        for pointY in datalist:
            dist = euclideanDistance(pointX, pointY)
            if dist == 0 :
                dist = 1.5
            dists.append(dist * -1)
        Similarity.append(dists)
    return Similarity


# #############################################################################
# AP演算法
def affinityPropagation(Similarity, lamda):
    #Responsibility = np.zeros_like(Similarity, dtype=np.float32)    #初始化 吸引矩陣
    #Availability = np.zeros_like(Similarity, dtype=np.float32)      #初始化 歸屬矩陣
    Responsibility = np.array(np.zeros_like(Similarity), dtype = 'float16')  # 初始化 吸引矩陣
    Availability = np.array(np.zeros_like(Similarity), dtype = 'float16')      #初始化 歸屬矩陣

    num = len(Responsibility)

    count = 0
    while count < 10:
        count+=1

        # update 吸引矩陣
        for Index in range(num):
            #print(len(Similarity[Index]))
            kSum = [s + a for s, a in zip(Similarity[Index], Availability[Index])]
            #print(kSum)
            for Kendex in range(num):
                kfit =  delete(kSum, Kendex)
                #print(fit)
                ResponsibilityNew = Similarity[Index][Kendex] - max(kfit)
                Responsibility[Index][Kendex] = lamda * Responsibility[Index][Kendex] + (1 - lamda) * ResponsibilityNew
                #print("Responsibility : %d" %Responsibility)

        # update 歸屬矩陣
        ResponsibilityT = Responsibility.T
        #print("ResponsibilityT : %d" %ResponsibilityT)

        for Index in range(num):
            iSum = [r for r in ResponsibilityT[Index]]

            for Kendex in range(num):
                #print(Kendex)
                #print("ddddddddddddd : ", Responsibility[Kendex] )
                ifit = delete(iSum, Kendex)
                ifit = filter(isNonNegative, ifit)  #上面 iSum 已經全部>0，會導致delete 下標錯誤

                #k == K 對角線的情況
                if Kendex == Index:
                    AvailabilityNew = sum(ifit)
                else:
                    result = Responsibility[Kendex][Kendex] + sum(ifit)
                    AvailabilityNew = result if result >0 else 0

                Availability[Kendex][Index] = lamda * Availability[Kendex][Index] + (1 - lamda) * AvailabilityNew
        print("#########################################################")
        print(Responsibility)
        print(Availability)
        print("#########################################################")
        return Responsibility + Availability


# #############################################################################
def computeCluster(fitable, data):
    clusters = {}
    num = len(fitable)
    for node in range(num):
        fit = list(fitable[node])
        key = fit.index(max(fit))
        #if not clusters.has_key(key):  .has_key()功能在python3x已經被移除
        if not clusters.__contains__(key):
            clusters[key] = []
        point = tuple(data[node])
        clusters[key].append(point)

    return clusters



# #############################################################################
def delete(lt, index):
    lt = lt[:index] + lt[index+1:]
    return lt

def isNonNegative(x):
    return x >= 0

# ---------------------------------------------------------------------------
#Similarity = np.zero((10000,10000), dtype = 'float32')

Similarity = computeSimilarity(data)

Similarity = np.array(Similarity)

print("Similarity",Similarity)
print("------------------------------------------------------------------------")
fitable = affinityPropagation(Similarity, 0.34)

print(fitable)

clusters = computeCluster(fitable,data)

#print(clusters)
# ---------------------------------------------------------------------------
clusters = clusters.values()

print(len(clusters))


# #############################################################################
def plotClusters(clusters, title):
    # 畫圖
    plt.figure(figsize = (8,5), dpi = 80)
    axes = plt.subplot(111)
    col = []
    r = lambda: random.randint(0,255)

    for index in range(len(clusters)):
        col.append(('#%02X%02X%02X' % (r(),r(),r())))
    color = 0
    for cluster in clusters:
        cluster = np.array(cluster).T
        axes.scatter(cluster[0], cluster[1], cluster[2], cluster[3], cluster[4], cluster[5], s = 20, c = col[color])
        color+= 1
    plt.title(title)
    #plt.show()


# #############################################################################
plotClusters(clusters, "clusters by affinity propagation")
plt.show()
#'''