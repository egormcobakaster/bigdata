import sys, os
SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')
from pyspark.ml import Pipeline, PipelineModel


test_path = sys.argv[2]
model_path = sys.argv[1]
pred_path = sys.argv[1]

schema = StructType([
    StructField("overall", FloatType()),
    StructField("vote", IntegerType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", IntegerType())
])

dataset = spark.read.json(test_path, schema=schema)

model = PipelineModel.load(model_path)
prediction = pipeline_model.transform(dataset)
prediction = prediction.withColumn("prediction", when(prediction.prediction > 5, 5)
                                   .when(prediction.prediction < 1, 1)
                                   .otherwise(prediction.prediction))
prediction.select("prediction").write.format("csv").save(pred_path, sep=',')

