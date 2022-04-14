
from pyspark.ml.feature import *
from pyspark.ml import Estimator, Transformer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline



stop_words = StopWordsRemover.loadDefaultStopWords("english")
tokenizer = Tokenizer(inputCol="reviewText", outputCol="words1")
swr = StopWordsRemover(inputCol="words1", outputCol="words_filtered", stopWords=stop_words)
word2Vec1 = Word2Vec(vectorSize=100, minCount=7, inputCol="words_filtered", outputCol="result")
lr = LinearRegression(featuresCol=word2Vec1.getOutputCol(), labelCol="overall", maxIter=50)
pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    word2Vec1,
    lr
])
