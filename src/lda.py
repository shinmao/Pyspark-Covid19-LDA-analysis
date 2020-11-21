from __future__ import print_function

import numpy as np
import os, json
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import ArrayType, StringType, DoubleType
from pyspark.sql.functions import udf, col, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.clustering import LDA

def termsIdx2Term(vocab):
    def termsIdx2Term(termIndices):
        return [vocab[int(index)] for index in termIndices]
    return udf(termsIdx2Term, ArrayType(StringType()))

def ith_(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None

if __name__ == "__main__":
    spark = SparkSession.builder.master("local").appName("COVID_LDA").getOrCreate()
    sc = spark.sparkContext

    path = '/user/hchen28/input/*.json'
    df = spark.read.json(path).cache()

    # tokenized
    regTokenizer = RegexTokenizer(inputCol="body_text", outputCol="cleaned_txt", pattern="\\W")
    regTokened = regTokenizer.transform(df)
    # remove stopword
    remover = StopWordsRemover(inputCol="cleaned_txt", outputCol="filtered")
    removed = remover.transform(regTokened)
    # token counts
    vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
    vecmodel = vectorizer.fit(removed)
    vectorized = vecmodel.transform(removed)
    # lda
    #tuned_topics = [8, 10, 12, 14, 16, 18, 20]
    tuned_topics = [9]
    for tuned_topic in tuned_topics:
        lda = LDA(k = tuned_topic, maxIter = 50, optimizer = "em")
        #pipeline = Pipeline(stages = [regTokenizer, remover, vectorizer, lda])

        ldamodel = lda.fit(vectorized)

        with open("/home/hchen28/tuned.txt", "a") as f:
            f.write("logLikelihood topic " + str(tuned_topic) + ": " + str(ldamodel.logLikelihood(vectorized)) + "\n")
            f.write("logPerplexity topic " + str(tuned_topic) + ": " + str(ldamodel.logPerplexity(vectorized)) + "\n")

        tops = ldamodel.describeTopics(10)

        vocab_model = vecmodel
        vocab_list = vocab_model.vocabulary
        final = tops.withColumn("Terms", termsIdx2Term(vocab_list)("termIndices"))
        final.show()

        # word distribution for each topic
        for i in final.collect():
            title = str(i['topic'])
            path = '/home/hchen28/result/' + title + '.txt'
            word = []
            prob = []
            for term in i['Terms']:
                word.append(term)
            for weight in i['termWeights']:
                prob.append(weight)
            with open(path, 'a+') as output:
                for i in word:
                    output.write(i + " ")
                for j in prob:
                    output.write(str(j) + " ")
                output.write("\n")
            x = list(range(0, 10, 1))
            plt.xticks(x, word)
            plt.plot(x, prob, color = 'b')
            plt.xlabel("word")
            plt.ylabel("probability")
            plt.title(title)
            filename = "/local-path/topic" + title + ".png"
            plt.savefig(filename)
            plt.show()

        transformed = ldamodel.transform(vectorized)
        transformed.show()

        ith = udf(ith_, DoubleType())
        explaineddf = transformed.select(["paper_id"] + [ith("topicDistribution", lit(i)).alias("topic_" + str(i)) for i in range(tuned_topic)])
        explaineddf.show()

        times = 0
        res_dist = "/home/hchen28/result/dist.txt"
        for paper in explaineddf.collect():
            paperid = paper["paper_id"]
            with open(res_dist, 'a+') as output:
                output.write(str(paperid))
                for i in range(tuned_topic):
                    idx = "topic_" + str(i)
                    topic_dist = paper[idx]
                    output.write(' ' + str(topic_dist))
                output.write('\n')
            times = times + 1
            if times > 5:
                break


    spark.stop()

