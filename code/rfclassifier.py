from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF,IDF, StringIndexer, CountVectorizer, VectorIndexer
from pyspark.ml.classification import RandomForestClassifier

def lr_train(data):
    #Logistic Regression using Count Vector Features
    (trainingData, testData) = data.randomSplit([0.9, 0.1], seed=100)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="cfeatures", vocabSize=10000, minDF=5)
    '''hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features",minDocFreq=5)'''
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0,featuresCol=countVectors.getOutputCol(), labelCol="label")
    pipeline = Pipeline(stages=[label_stringIdx,countVectors,lr])
    pipelineFit = pipeline.fit(trainingData)
    predictions = pipelineFit.transform(testData)
    #predictions.show(5)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    #evaluator.evaluate(predictions)
    return evaluator.evaluate(predictions),lr

def lr2_train(data):
    print data.columns
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
    label_stringIdx = StringIndexer(inputCol="_c0", outputCol="label")
    pipeline = Pipeline(stages=[countVectors,label_stringIdx])
    # Fit the pipeline to training documents.
    pipelineFit = pipeline.fit(data)
    dataset = pipelineFit.transform(data)
    dataset.show(5)
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)

    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    evaluator.evaluate(predictions)
    return evaluator.evaluate(predictions), lrModel

def rf_train(data):
    #Random Forest Classifier
    (trainingData, testData) = data.randomSplit([0.9, 0.1], seed = 100)
    countVectors = CountVectorizer(inputCol = "filtered", outputCol = "rfFeatures", vocabSize = 10000, minDF = 5)
    label_stringIdx = StringIndexer(inputCol = "_c0", outputCol = "label")
    rf = RandomForestClassifier(labelCol = "label", featuresCol = "rfFeatures", maxMemoryInMB = 16, numTrees = 100)
    pipeline = Pipeline(stages = [label_stringIdx,countVectors,rf])
    pipelineFit = pipeline.fit(trainingData)
    predictions = pipelineFit.transform(testData)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    return evaluator.evaluate(predictions), rf

def nb_train(data):
    #Naive Bayes Classifier
    return

'''hashingTF = HashingTF()
tf = hashingTF.transform(data.rdd)
# While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
# First to compute the IDF vector and second to scale the term frequencies by IDF.
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
print tfidf.columns
idfIgnore = IDF(minDocFreq=2).fit(tf)
tfidfIgnore = idfIgnore.transform(tf)
data=data.withColumn("tfidf",tfidf)

print predictions.filter(predictions['prediction'] == 0)\
    .select("filtered", "_c0", "probability", "label", "prediction") \
    .orderBy("probability", ascending=False) \
    .show(n=10, truncate=30)

rf = RandomForestClassifier(labelCol="_c0", \
                            featuresCol="filtered", \
                            numTrees=100, \
                            maxDepth=4, \
                            maxBins=32)
# Train model with Training Data
rfModel = rf.fit(trainingData)
predictions = rfModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("_c0", "filtered", "prediction") \
    .show(n=10, truncate=30)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(data)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.select("_c0", "features").show()

labelIndexer = StringIndexer(inputCol="_c0", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = \
    VectorIndexer(inputCol="filtered", outputCol="indexedFeatures", maxCategories=4).fit(data)
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only'''