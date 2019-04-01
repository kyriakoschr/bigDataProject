from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()

df = spark.read.csv("train.csv",header=True,sep="\t");

print(df.collect(1))