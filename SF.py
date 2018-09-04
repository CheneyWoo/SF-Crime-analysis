#import the data reader
from csv import reader

#prepare data 
df_crimes = crime_data_lines.map(lambda line: [x.strip('"') for x in next(reader([line]))])

#get header
header = df_crimes.first()

header
Out[159]: 
['IncidntNum',
 'Category',
 'Descript',
 'DayOfWeek',
 'Date',
 'Time',
 'PdDistrict',
 'Resolution',
 'Address',
 'X',
 'Y',
 'Location',
 'PdId']
 
 #remove the first line of data
crimes = df_crimes.filter(lambda x: x != header)

#approach 1: use RDD 
#approach 2: use Dataframe, register the RDD to a dataframe // crimeDF = crimes.map(lambda p: Row(IncidntNum=p[0], Category=int(p[1])))
#approach 3: use SQL 
# 1st question: 
#Write a Spark program that counts the number of crimes for different category.
print crimes.count()
category = crimes.take(crimes.count())
category = crimes.map(lambda x: (x[1],1))
category.countByKey().items()
rddformdata = sc.parallelize(category.countByKey().items())
categorysorted = rddformdata.sortBy(lambda a:a[1])
categorysorted.collect()

8977
Out[163]: 
[('SEX OFFENSES, NON FORCIBLE', 1),
 ('PORNOGRAPHY/OBSCENE MAT', 1),
 ('TREA', 1),
 ('FAMILY OFFENSES', 1),
 ('EXTORTION', 2),
 ('SUICIDE', 3),
 ('LOITERING', 4),
 ('LIQUOR LAWS', 4),
 ('EMBEZZLEMENT', 5),
 ('BRIBERY', 7),
 ('KIDNAPPING', 10),
 ('DRIVING UNDER THE INFLUENCE', 16),
 ('DISORDERLY CONDUCT', 18),
 ('DRUNKENNESS', 19),
 ('PROSTITUTION', 20),
 ('FORGERY/COUNTERFEITING', 29),
 ('ARSON', 29),
 ('RUNAWAY', 39),
 ('SEX OFFENSES, FORCIBLE', 40),
 ('RECOVERED VEHICLE', 48),
 ('STOLEN PROPERTY', 66),
 ('WEAPON LAWS', 106),
 ('TRESPASS', 109),
 ('SECONDARY CODES', 118),
 ('FRAUD', 155),
 ('DRUG/NARCOTIC', 167),
 ('ROBBERY', 187),
 ('MISSING PERSON', 265),
 ('BURGLARY', 302),
 ('WARRANTS', 312),
 ('SUSPICIOUS OCC', 312),
 ('VEHICLE THEFT', 353),
 ('VANDALISM', 650),
 ('ASSAULT', 780),
 ('NON-CRIMINAL', 991),
 ('OTHER OFFENSES', 1002),
 ('LARCENY/THEFT', 2805)]
 
 ####2nd question
##### Write a program that counts the number of crimes for different district
district = crimes.take(crimes.count())
district = crimes.map(lambda x: (x[0:][6],1))
district.countByKey().items()
rddformdata2 = sc.parallelize(category.countByKey().items())
districtsorted = rddformdata2.sortBy(lambda a:a[1])
districtsorted.collect()

Out[164]: 
[('SEX OFFENSES, NON FORCIBLE', 1),
 ('PORNOGRAPHY/OBSCENE MAT', 1),
 ('TREA', 1),
 ('FAMILY OFFENSES', 1),
 ('EXTORTION', 2),
 ('SUICIDE', 3),
 ('LOITERING', 4),
 ('LIQUOR LAWS', 4),
 ('EMBEZZLEMENT', 5),
 ('BRIBERY', 7),
 ('KIDNAPPING', 10),
 ('DRIVING UNDER THE INFLUENCE', 16),
 ('DISORDERLY CONDUCT', 18),
 ('DRUNKENNESS', 19),
 ('PROSTITUTION', 20),
 ('FORGERY/COUNTERFEITING', 29),
 ('ARSON', 29),
 ('RUNAWAY', 39),
 ('SEX OFFENSES, FORCIBLE', 40),
 ('RECOVERED VEHICLE', 48),
 ('RECOVERED VEHICLE', 48),
 ('STOLEN PROPERTY', 66),
 ('WEAPON LAWS', 106),
 ('TRESPASS', 109),
 ('SECONDARY CODES', 118),
 ('FRAUD', 155),
 ('DRUG/NARCOTIC', 167),
 ('ROBBERY', 187),
 ('MISSING PERSON', 265),
 ('BURGLARY', 302),
 ('WARRANTS', 312),
 ('SUSPICIOUS OCC', 312),
 ('VEHICLE THEFT', 353),
 ('VANDALISM', 650),
 ('ASSAULT', 780),
 ('NON-CRIMINAL', 991),
 ('OTHER OFFENSES', 1002),
 ('LARCENY/THEFT', 2805)]
 
 #### 3rd question
##### Write a program to count the number of crimes each Sunday at SF downtown. 
###### hints: define your spatial function for filtering data
crimesonsunday = crimes.filter(lambda x: x[0:][3] == 'Sunday')
crimesonsundayflt = crimesonsunday.filter(lambda x: x[0:][9]+x[0:][10] >= -84)
dis = crimesonsundayflt.map(lambda x: (x[0:][4],1))
sorted(dis.countByKey().items())
Out[165]: [('07/16/2017', 405), ('07/23/2017', 442), ('07/30/2017', 391)]

##### Extra: visualize the spatial distribution of crimes and run a kmeans clustering algorithm
import numpy as np
import matplotlib.pyplot as plt

x = crimes.map(lambda x:x[9]).collect()
y = crimes.map(lambda x:x[10]).collect()

from pandas import *
from ggplot import *
pydf = DataFrame({'x':x,'y':y})
p = ggplot(pydf,aes('x','y')) + \
    geom_point(color = 'blue')
display(p)

from numpy import array
from math import sqrt

from pyspark.mllib.clustering import KMeans, KMeansModel

# Load and parse the data

datasplit1 = crimes.map(lambda x: (x[9],x[10]))
#print datasplit1.first()[0]
datasplit = datasplit1.map(lambda line: array([float(line[0]),float(line[1])]))
#print type(datasplit)
# Build the model (cluster the data)
clusters = KMeans.train(datasplit, 4, maxIterations=10, initializationMode="random")
clusters.clusterCenters
print clusters.clusterCenters
#print type(clusters.clusterCenters)
#print type(clusters.clusterCenters[0])
#Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = datasplit.map(lambda point: error(point)).reduce(lambda x, y: x + y)
#center1 = clusters.centers[clusters.predict(array([-122.44383721,   37.77953436]))]
#print center1
print("Within Set Sum of Squared Error = " + str(WSSSE))
# Save and load model
#clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
#sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")

[array([-122.41187284,   37.79039618]), array([-122.41816633,   37.76877959]), array([-122.47403813,   37.75457395]), array([-122.40955324,   37.73028157])]
Within Set Sum of Squared Error = 148.31068216

rddcenter = sc.parallelize(mycenters)
centerx = rddcenter.map(lambda x: x[0]).collect()
centery = rddcenter.map(lambda x: x[1]).collect()
print centery

class Scatter:
  def __init__(self, num):
    self.num = num
    self.scatter = datasplit.filter(lambda x: (clusters.centers[clusters.predict(x)] == mycenters[num].tolist())[0])
    self.xx = self.scatter.map(lambda x:x[0]).collect()
    self.yy = self.scatter.map(lambda x:x[1]).collect()
  def graph(self):
    #self.graph = DataFrame({'x':self.xx,'y':self.yy})
    return DataFrame({'x':self.xx,'y':self.yy})

scatter1 = Scatter(0)
scatter2 = Scatter(1)
scatter3 = Scatter(2)
scatter4 = Scatter(3)

rddcenter = sc.parallelize(mycenters)
centerx = rddcenter.map(lambda x: x[0]).collect()
centery = rddcenter.map(lambda x: x[1]).collect()
graphcenter = DataFrame({'x':centerx,'y':centery})

pp3 = ggplot(scatter1.graph(), aes('x','y')) + geom_point(scatter1.graph(), color = 'blue') + geom_point(scatter2.graph(), color = 'gray') + geom_point(scatter3.graph(), color = 'yellow') + geom_point(scatter4.graph(), color = 'green') + geom_point(graphcenter,shape = '*', color = 'red', size = 300)
display(pp3)

#print scatter.graph()
