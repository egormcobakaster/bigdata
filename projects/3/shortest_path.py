#!/opt/conda/envs/dsenv/bin/python
import os
import sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Pagerank").getOrCreate()


from pyspark.sql.types import *
from pyspark.sql.functions import *


v_from = sys.argv[1]
v_to = sys.argv[2]
data = sys.argv[3]
path = sys.argv[4]

schema_1 = StructType([
    StructField("to", IntegerType(), False),
    StructField("from1", IntegerType(), False)
])
schema_dist = StructType([
    StructField("v", IntegerType(), False),
    StructField("distance", IntegerType(), False)
])


edges = spark.read.csv(data, sep="\t", schema=schema_1)
edges.cache()

distances = spark.createDataFrame([(v_from, 0)], schema_dist)
n = 0

while True:
    candidates = (distances
                  .join(edges, distances.v==edges.from1)                   
                 )

    candidates = candidates.drop('v')
    candidates = candidates.withColumnRenamed('distance','dist')
    candidates = candidates.withColumnRenamed('to','v1')
    candidates = candidates.withColumnRenamed('from1','{}'.format(n))
    candidates = candidates.withColumn('dist', col('dist') + lit(1))
    candidates.show()
    print('{}'.format(n))
    new_distances = (distances
                     .join(candidates, distances.v==candidates.v1, how="full_outer")
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
    distances = distances.withColumn("tmp", distances.v)
    distances = distances.withColumn("v", distances['{}'.format(n-1)])
    distances = distances.withColumn('{}'.format(n-1), distances.tmp)
    distances = distances.drop("tmp")
    distances = distances.where(distances['{}'.format(n-1)]==v_to)
    distances.write.format("csv").save(path, sep=',')
