# Assignment - 4
# Submitted by Rucha Kulkarni

# importing libraries
import pyspark as spark
import sys
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.feature import HashingTF

# Setting frequency to read first 500 words of the input where each feature is a word
sc = spark.SparkContext(appName="SpamDetection")
termFreq = HashingTF(numFeatures = 500)

# Function to read Spam email and create RDD
def getSpamInputData(spamEmailTextPath):
    spamEmailRDD = sc.textFile(spamEmailTextPath)
    return spamEmailRDD

# Function to read nonSpam email and create RDD
def getNonSpamInputData(nonSpamEmailTextPath):
    nonSpamEmailRDD = sc.textFile(nonSpamEmailTextPath)
    return nonSpamEmailRDD

# Function to read query data and create RDD
def getQueryData(queryDataTextPath):
    queryDataRDD = sc.textFile(queryDataTextPath)
    return queryDataRDD

# Function to build prediction model using Logistic Regression
def buildModel(spamEmailRDD, nonSpamEmailRDD):
    spamData = trainSpamData (spamEmailRDD)
    nonSpamData = trainNonSpamData (nonSpamEmailRDD)
    trainingData = spamData.union(nonSpamData)
    trainingData.cache()
   # using in-built library to train the model
    model = LogisticRegressionWithSGD.train(trainingData)
    return model

# Function to train the model with Spam data
def trainSpamData(spamEmailRDD):
    spamRDD = spamEmailRDD.map(lambda email: termFreq.transform(email.split(" ")))
    classifySpamRDD = spamRDD.map(lambda features: LabeledPoint(1, features))
    return classifySpamRDD

# Function to train the model with nonSpam data
def trainNonSpamData(nonSpamEmailRDD):
    nonSpamRDD = nonSpamEmailRDD.map(lambda email: termFreq.transform(email.split(" ")))
    classifyNonSpamRDD = nonSpamRDD.map(lambda features: LabeledPoint(0, features))
    return classifyNonSpamRDD

# Function to predict and create RDD <classification, email>
def predictData(model, queryDataRDD):
    predictionMap = queryDataRDD.map(lambda email: (model.predict(termFreq.transform(email.split(" "))), email))
    return predictionMap

# Print the output with classification for Query data
def printOutput(outPutMap):
    print(outPutMap.collect())

# Function to count the number of Spam data classifications
def countSpamAccuracy(outPutRDD):
    totalCount = outPutRDD.count()
    spamAccuracyCount = 0
    for (classification, email) in outPutRDD.collect():
        if classification == 1:
            spamAccuracyCount = spamAccuracyCount + 1
    return spamAccuracyCount

# Function to count the number of nonSpam data classifications
def countNonSpamAccuracy(outPutRDD):
    totalCount = outPutRDD.count()
    nonSpamAccuracyCount = 0
    for (classification, email) in outPutRDD.collect():
        if classification == 0:
            nonSpamAccuracyCount = nonSpamAccuracyCount + 1
    return nonSpamAccuracyCount

# Function to calculate overall accuracy of the model using Spam and nonSpam input data
def calculateOverAllScore(outPutSpamMapRDD, outPutNonSpamMapRDD):
    totalCount = outPutSpamMapRDD.count() + outPutNonSpamMapRDD.count()
    spamAccuracyCount = countSpamAccuracy(outPutSpamMapRDD)
    nonSpamAccuracyCount = countNonSpamAccuracy(outPutNonSpamMapRDD)
    overallAccuracyScore = ((spamAccuracyCount + nonSpamAccuracyCount) / totalCount) * 100
    return overallAccuracyScore

# Check if the number of args is 4. If not, exit()
if(len(sys.argv) != 4) :
    print("Please provide exactly 3 inputs : <Path_to_spamfile> <Path_to_nonspamfile> <Path_to_queryfile>")
    exit()

# Get input text files from user
spamEmailTextPath = sys.argv[1]
nonSpamEmailTextPath = sys.argv[2]
queryDataTextPath = sys.argv[3]

# Invoke functions to build the model and do the necessary accuracy calculation
spamEmailRDD = getSpamInputData(spamEmailTextPath)
nonSpamEmailRDD = getNonSpamInputData(nonSpamEmailTextPath)
queryDataRDD = getQueryData(queryDataTextPath)

model = buildModel(spamEmailRDD, nonSpamEmailRDD)
outPutMapRDD = predictData(model, queryDataRDD)
print( outPutMapRDD.collect())

outPutSpamMap = predictData(model, spamEmailRDD)
outPutNonSpamMap = predictData(model, nonSpamEmailRDD)
spamAccuracyScore = (countSpamAccuracy(outPutSpamMap) / outPutSpamMap.count()) * 100
nonSpamAccuracyScore = (countNonSpamAccuracy(outPutNonSpamMap) / outPutNonSpamMap.count()) * 100

overallAccuracy = calculateOverAllScore(outPutSpamMap, outPutNonSpamMap)

print 'Spam Accuracy of Model (%) is:',spamAccuracyScore
print 'NonSpam Accuracy of Model (%) is:',nonSpamAccuracyScore
print 'Overall Accuracy of Model (%) is:',overallAccuracy

