from enum import unique
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

first = pd.read_csv('storage/data_with_groups.csv')

first.group.unique()

first_noduplicates = first.groupby(['REGNUM', 'date', 'group']).sum().reset_index()

first_noduplicates

#make overall consumption
first_overall_consuption = first.groupby(['date', 'group']).sum().reset_index()[['date', 'group', 'PRC_AMT']]


#make for food
df_wide_food=pd.pivot(first_noduplicates[first_noduplicates['group']== 'food'], index=['date'], columns = 'REGNUM',values = 'PRC_AMT').fillna(0) #Reshape from long to wide


#plot for all categories
for group in first_overall_consuption['group'].unique().tolist():
    if str(group) != 'money':
        plt.plot(first_overall_consuption[first_overall_consuption['group']==str(group)]['PRC_AMT'].rolling(window=20).mean(), label = str(group))
plt.legend()    
plt.show()



def Norm01(x):
    mi=np.nanmin(x)
    ma=np.nanmax(np.array(x)-mi)
    if ma>0.:
        x_n=(np.array(x)-mi)/ma
        return x_n, mi, ma
    else:
        return np.zeros(len(x)), mi, ma
    
def  MovingAverage(x, numb=10):
    n=len(x)//numb
    ma=list(x[:n])
    for j in range(len(x)-n):
        ma.append(np.mean(x[j:j+n]))
    return np.array(ma)


#plot food and all customers

plt.plot(MovingAverage(Norm01(first_overall_consuption[first_overall_consuption['group']=="food"]['PRC_AMT'])[0], numb = 20)\
    ,label = 'food'\
    ,linewidth=2, color='red')

for customer in df_wide_food.columns.tolist()[0:100]:
    plt.plot(MovingAverage(Norm01(df_wide_food[customer])[0], numb = 20), alpha=0.3, color = 'blue')

plt.legend()
plt.show()


plt.plot(df_wide_food[df_wide_food.columns.tolist()[100]])


df_wide_food_smoothed = df_wide_food.apply(lambda x : MovingAverage(Norm01(x)[0], numb = 20))




from tslearn.clustering import TimeSeriesKMeans

df_wide_food_smoothed_thin = df_wide_food_smoothed.sample(frac=0.20,axis='columns')

model = TimeSeriesKMeans(n_clusters=5, metric="dtw",
                         max_iter=10, random_state=2022, verbose=1)

clustered = model.fit(np.array(df_wide_food_smoothed_thin.transpose()))

array_data = np.array(df_wide_food_smoothed_thin.transpose())

clusters = model.predict(np.array(df_wide_food_smoothed_thin.transpose()))

from collections import Counter

len(clusters)
Counter(clusters)



def indexes(iterable, obj):
    return (index for index, elem in enumerate(iterable) if elem == obj)


for group in range(5):
    
    temp = array_data[list(indexes(clusters.tolist(), group))]
    series = np.mean(temp,axis=0)
    plt.plot(series, label = str(group))

#plt.plot(MovingAverage(Norm01(first_overall_consuption[first_overall_consuption['group']=="food"]['PRC_AMT'])[0], numb = 20)\
#    ,label = 'food'\
#    ,linewidth=2, color='red')

plt.axvline(x=87, color='b', ls='--', alpha=0.5, label='crisis')
plt.legend()
plt.show()


strange_cluster = array_data[list(indexes(clusters.tolist(), 4))]
for i in range(len(strange_cluster)):
    plt.plot(strange_cluster[i])
plt.show()

plt.plot(strange_cluster[10])



#trying to cut 


df_wide_food_smoothed_cut = df_wide_food_smoothed[50:150]

model = TimeSeriesKMeans(n_clusters=5, metric="dtw",
                         max_iter=10, random_state=2022, verbose=1)

clustered_cut = model.fit(np.array(df_wide_food_smoothed_cut.transpose()))

array_data_cut = np.array(df_wide_food_smoothed_cut.transpose())

clusters_cut = model.predict(np.array(df_wide_food_smoothed_cut.transpose()))

from collections import Counter

len(clusters_cut)
Counter(clusters_cut)



def indexes(iterable, obj):
    return (index for index, elem in enumerate(iterable) if elem == obj)



groups = []
for group in range(5):
    
    temp = array_data_cut[list(indexes(clusters_cut.tolist(), group))]
    series = np.mean(temp,axis=0)
    plt.plot(series, label = str(group))
    groups.append(series)

plt.plot(MovingAverage(Norm01(first_overall_consuption[first_overall_consuption['group']=="food"]['PRC_AMT'][50:150])[0], numb = 20)\
    ,label = 'food'\
    ,linewidth=2, color='red')

plt.axvline(x=40, color='b', ls='--', alpha=0.5, label='crisis')
plt.legend()
plt.show()

strange_cluster_cut = array_data_cut[list(indexes(clusters_cut.tolist(), 4))]

for i in range(len(strange_cluster_cut)):
    plt.plot(strange_cluster_cut[i])

plt.show()

plt.plot(strange_cluster_cut[30])


#Tests

from Libraries.pymssa import MSSA

import statsmodels.api as sm
import Libraries
from Libraries.Autoregr import VARModel
from Libraries.Util import Norm01
from Libraries.Util import Nback
from Libraries.Util import Metr
from Libraries.Util import MovingAverage

def VARTest(x,y,maxlag=52):
    b=len(x)-6
    fwd=6
    score=200
    lag=0    
    for l in range(maxlag):
        x1,mi,ma=Norm01(x)
        x1=x1[l:]
        y1,_,_=Norm01(y)
        y1=pd.Series(y1).shift(l)[l:]
        vec=pd.DataFrame({'reg':x1, 'prd':y1})
        x_test=Nback(VARModel(vec[:b],fwd), mi,ma)
        d = Metr(x[b:b+fwd], x_test)
        if d[2]<score:
            score=d[2]
            lag=l
    return score, lag


food_overall = MovingAverage(Norm01(first_overall_consuption[first_overall_consuption['group']=="food"]['PRC_AMT'][50:150])[0], numb = 20)

data = pd.DataFrame(np.array(groups).T)
data.columns = [str(x) for x in list(data.columns)]
data['overall'] = food_overall

data.to_csv('storage/5groups.csv')

i = 0
for y in list(data.columns)[0:5]:
    print('group - ', i , 'score = ' ,VARTest(data[str(y)],data['overall'])[0], 'lag = ',\
        VARTest(data[str(y)],data['overall'])[1])
    i+=1

from statsmodels.tsa.vector_ar.var_model import VAR











def CrossCorr(datax, datay, maxlag=52):
    ccor=0
    lag=0
    dx=pd.Series(datax)
    dy=pd.Series(datay)
    for i in range(1,maxlag):
        c=abs(dx.corr(dy.shift(i),method='spearman'))
        if c>ccor:
            ccor=c
            lag=i
    return lag,ccor 

i = 0
for y in list(data.columns)[0:5]:
    print('group - ', i , 'lag = ' ,CrossCorr(data[str(y)],data['overall'])[0], 'ccor = ',\
        CrossCorr(data[str(y)],data['overall'])[1])
    i+=1

test_list = list(range(0,5))
#by pairs 
pairs = [(a, b) for idx, a in enumerate(test_list) for b in test_list[idx + 1:]]

for x in range(5):
    for y in range(5):
        if x != y:
            pair = [x,y]
            print('group-', pair[0], "-", pair[1] , 'lag = ' ,CrossCorr(data[str(pair[0])],data[str(pair[1])])[0], 'ccor = ',\
                CrossCorr(data[str(pair[0])],data[str(pair[1])])[1])
        


#by one vs others
x = 2
for x in range(5):
    cols  =list(data.columns)[:5]
    cols.pop(x)
    cols = [str(x) for x in cols]
    series = data[cols].mean(axis=1)
    print('group-', x , 'lag = ' ,CrossCorr(data[str(x)],series)[0], 'ccor = ',\
        CrossCorr(data[str(x)],series)[1])
        
