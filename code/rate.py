import pyspark.sql.functions as func
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, regexp_replace, split
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from rfclassifier import lr_train
import nltk
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  lowercased_str = x.lower()
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, '')
  return lowercased_str


'''def remove_extra_ws(x):  #alternative method to cut extra whitespaces 1/3
  return " ".join(x.split())'''

def rate_transform(x):
    if ((x=="1") or (x=="2") or (x=="3")):
        x="0"
    elif ((x=="4") or (x=="5")):
        x="1"
    return x

def parse_data(path):
    spark = SparkSession.builder.appName("Parsing and removing stopwords").getOrCreate()
    if (path == "../train.csv"):
        lcs = udf(lower_clean_str)
        # rews=udf(remove_extra_ws) #2/3
        df = spark.read.csv(path, header=False, sep="\t");
        df = df.withColumn("_c1", lcs("_c1"))
        # df=df.withColumn("_c1",rews("_c1")) #3/3
        expres = [split(col("_c1"), " ").alias("_c1")]
        df = df.withColumn("_c1",*expres)
        remover = StopWordsRemover(inputCol="_c1", outputCol="filtered")
        swlist = remover.getStopWords()
        swlist= swlist + list(set(stopwords.words('english')))+ ['']
        remover.setStopWords(swlist)
        #remover.transform(df).select("filtered")

        final = remover.transform(df.select("_c1"))
        df = df.withColumn('row_index', func.monotonically_increasing_id())
        final = final.withColumn('row_index', func.monotonically_increasing_id())
        final = final.join(df["row_index", "_c0"], on=["row_index"]).drop("row_index").drop("_c1")
        # fdf.show()
        return final

    elif (path == "../test.csv"):
        lcs = udf(lower_clean_str)
        # rews=udf(remove_extra_ws) #2/3
        df = spark.read.csv(path, header=False, sep="\t");
        df = df.withColumn("_c0", lcs("_c0"))
        # df=df.withColumn("_c1",rews("_c1")) #3/3
        expres = [split(col("_c0"), " ").alias("_c0")]
        df = df.select(*expres)
        remover = StopWordsRemover(inputCol="_c0", outputCol="filtered")
        swlist = remover.getStopWords()
        swlist= swlist + list(set(stopwords.words('english')))+ ['']
        remover.setStopWords(swlist)
        remover.transform(df).select("filtered").show(1, truncate=False)
        final = remover.transform(df.select("_c1"))
        df = df.withColumn('row_index', func.monotonically_increasing_id())
        final = final.withColumn('row_index', func.monotonically_increasing_id())
        final = final.join(df["row_index", "_c0"], on=["row_index"]).drop("row_index").drop("_c1")
        return final

    else:
        print "Wrong File or Path"
        return -1

def main():
    rt = udf(rate_transform)
    path = sys.argv[1]
    df = parse_data(path)
    df = df.withColumn("_c0", rt("_c0"))
    df.show()
    (acc_lr,lr) =lr_train(df)
    print acc_lr
    print lr

if __name__=="__main__":
    main()
