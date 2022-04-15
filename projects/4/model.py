from pyspark.ml.feature import *
from pyspark.ml import Estimator, Transformer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline


stop_words = StopWordsRemover.loadDefaultStopWords("english")
tokenizer = Tokenizer(inputCol="reviewText", outputCol="words1")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words_filtered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="word_vector")
lr = LinearRegression(featuresCol=count_vectorizer.getOutputCol(), labelCol="overall", maxIter=50)


pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer,
    lr
])
