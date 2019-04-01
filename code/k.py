from pyspark.sql.functions import udf
from pyspark.sql.functions import col, regexp_replace, split
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover

def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
  lowercased_str = x.lower()
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, '')
  return lowercased_str

lcs=udf(lower_clean_str)
spark = SparkSession.builder.appName("Parsing and removing stopwords").getOrCreate()
df = spark.read.csv("../train.csv",header=False,sep="\t");
df=df.withColumn("_c1",lcs("_c1"))
expres = [split(col("_c1")," ").alias("_c1")]
df=df.select(*expres)
df.withColumn
remover = StopWordsRemover(inputCol="_c1", outputCol="filtered")
remover.transform(df).select("filtered").show(1,truncate=False)