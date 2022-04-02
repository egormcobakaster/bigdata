#!/opt/conda/envs/dsenv/bin/python
import os
import sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"

os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME
os.environ["PYSPARK_DRIVER_PYTHON"]= "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_EXECUTOR_PYTHON"]= "/opt/conda/envs/dsenv/bin/python"
PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf
from pyspark.sql import SparkSession

conf = SparkConf()

spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()

from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np
import pandas as pd

v_from = sys.argv[1]
v_to = sys.argv[2]
data = sys.argv[3]
path = sys.argv[4]

schema_graph = StructType([
    StructField("to", IntegerType(), False),
    StructField("from", IntegerType(), False)
])
schema_dist = StructType([
    StructField("v", IntegerType(), False),
    StructField("distance", IntegerType(), False)
])


edges = spark.read.csv(data, sep="\t", schema=schema_graph)
edges.cache()

distances = spark.createDataFrame([(v_from, 0)], schema_dist)
n = 0

while True:
    candidates = (distances
                  .join(edges, distances.v==edges.from)                   
                 )

    candidates = candidates.drop('v')
    candidates = candidates.withColumnRenamed('distance','dist')
    candidates = candidates.withColumnRenamed('to_v','v1')
    candidates = candidates.withColumnRenamed('from_v','{}'.format(n))
    candidates = candidates.withColumn('dist', col('dist') + lit(1))


    new_distances = (distances
                     .join(candidates, distances.v==candidates.v, how="full_outer")
                     .select("v1",
                             when(
                                 distances.distance.isNotNull(), distances.distance
                             ).otherwise(
                                 candidates.dist
                             ).alias("dist"))
                    ).persist()

    count = new_distances.where(new_distances.dist==n+1).count()

    if count > 0:
        n += 1            
        distances = candidates
        distances = distances.withColumnRenamed('v1','v')
        distances = distances.withColumnRenamed('dist','distance')
    else:
        break  

    target = (new_distances
              .where(new_distances.v1 == v_to)
             ).count()

    if  target > 0:
        break
    if n>1000:
        break
if n<1000:
    distances.where(distances.v == v_to)
    for row in Best.collect():
        mas = row
        break
    mas1 = list(mas[1:len(mas)])
    best_way = np.array(mas1[0:len(mas1)-2])
    print(best_way)
    best_way = np.append(best_way,(mas1[len(mas1)-2]), (mas1[len(mas1)-1]))
    df = pd.DataFrame(best_way.reshape(1,-1))
    df.to_csv(path, sep=',', index=False, header=False)
