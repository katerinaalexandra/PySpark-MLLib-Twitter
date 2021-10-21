from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import explode_outer, explode
from pyspark.sql.types import IntegerType


from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Normalizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.recommendation import ALS

import argparse


## mapPartitions function that reformats data into a key-value tuple where the key is the user_id
## and the value is a tuple consisting of the normalized TF vector and normalized CV vector.
def reformat_partition(list_of_records):
    final_iterator=[]
    for record in list_of_records:
        user_id=record["user_id"]
        vector_tfidf=record["features_norm"]
        vector_cv=record["cvfeatures_norm"]
        final_iterator.append((user_id,(vector_tfidf,vector_cv)))
    return iter(final_iterator)


## Using the broadcasted lookup table, this function takes the input user_id, extracts their TF and CV vector from the lookup table
## Then for each user in the lookup table, the function gets the dot product between the original user vector and each iterative user vector.
def get_vector_info(record):
    vector_info=vector_lookup_broadcast.value.get(record)

    tf_sim_list=[]
    cv_sim_list=[]

    for key,value in vector_lookup_broadcast.value.items():
        if key != record:
            temp_user_id=key
            temp_vector_info=value

            tf_sim=vector_info[0].dot(temp_vector_info[0])
            cv_sim=vector_info[1].dot(temp_vector_info[1])

            tf_sim_list.append((tf_sim,temp_user_id))
            cv_sim_list.append((cv_sim,temp_user_id))


    return(record, (tf_sim_list,cv_sim_list))


## mapPartition function to reformat the data in workload 2. Returns a tuple of (user_id, hashed user_id, mention_user_id, hashed mention_user_id)
## Used in the broadcast lookup table to convert hashed values into original values.
def reformat_w2_partition(list_of_records):
    final_iterator=[]
    for record in list_of_records:
        user_id=record[0]
        hash_user_id=record[1]
        mention_user_id=record[2]
        hash_mention_user_id=record[3]
        final_iterator.append((user_id, hash_user_id, mention_user_id, hash_mention_user_id))
    return iter(final_iterator)


## mapValues function to reformat the values in workload 2.

def reformat_values_w2(record):
    try:
        id1=record[0][0]
        id2=record[1][0]
        id3=record[2][0]
        id4=record[3][0]
        id5=record[4][0]
        return(id1, id2, id3, id4, id5)
    except:
        return ()

    
## Use lookup table to convert mention user hashes into original mentioner user ID.
## For each of top 5 recommendations, this function runs and returns original ID.
def convert_mention(mention_hashes):
    try:
        return_list=[]
        for mention_hash in mention_hashes:
            for value in id_hash_map.value:
                if value[3] == mention_hash:
                    mention_id=value[2]
                    return_list.append(mention_id)
                    break
        return return_list
    except:
        return()

## Use lookup table to convert user hashes into original user ID
def convert_user(user_hash):
    try:
        for value in id_hash_map.value:
            if value[1] == user_hash:
                return value[0]
    except:
        return ()


## This is the function that calls the convert_user and convert_mention functions
## Extracts the user hash and mention user hash from the recommender and calls 2 separate functions
def convert_hash(record):
    try:
        user_hash=record[0]
        mention_hashes=record[1]

        user_id=convert_user(user_hash)

        mention_ids=convert_mention(mention_hashes)

        return(user_id, mention_ids)

    except:
        return()

## ===================================================================================================================== ##

if __name__ == "__main__":
    spark=SparkSession.builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").appName("Assignment 2").getOrCreate()

    sc=spark.sparkContext

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--output", help="the output path",
                        #default='tweets_out_script')

    #parser.add_argument("--user_id", help="the user_id to calculate cosine similarities for",
                        #default=202170318)
    #args = parser.parse_args()
    #output_path = args.output
    test_id=202170318
    #test_id=args.user_id


    ## Read in full JSON file and cache for use in multiple workloads.
    tweets=spark.read.option("multiline","true").json("tweets.json").cache()

    ## Workload 1
    ## Extract relevant data - user_id, retweet_id and replyto_id;
    ## Ignore records where the tweet is neither a reply nor a retweet;
    ## Concatenate replyto_id and retweet_id into one column, as a tweet cannot be both;
    ## Group by user_id and concatenate grouped values into a list for document representation;
    ## Cache resulting dataframe.
    tweets_w1=tweets.filter("replyto_id IS NOT NULL OR retweet_id IS NOT NULL").select('user_id', f.concat_ws('-', 'replyto_id','retweet_id').alias("tweet_id")).groupBy("user_id").agg(f.concat_ws(", ", f.collect_list("tweet_id")).alias("ids")).cache()


    ## Feature Extraction
    ## Both extractors require tokenized document
    tokenizer = Tokenizer(inputCol="ids", outputCol="ids_tokenized")
    tokenized_data = tokenizer.transform(tweets_w1).cache()
    
    ## Apply CV algorithm for CV feature extraction
    cv=CountVectorizer(inputCol='ids_tokenized', outputCol="cvfeatures", vocabSize=200, minDF=1)
    cvmodel=cv.fit(tokenized_data)
    cv_data=cvmodel.transform(tokenized_data)
    
    ## Continue with TF-idf
    ## HashingTF - transformer which takes set of terms and converts into fixed-length feature vectors
    hashingTF = HashingTF(inputCol="ids_tokenized", outputCol="rawFeatures", numFeatures=200)
    featurized_data = hashingTF.transform(cv_data).cache()
    
    ## Apply idf estimator to hashed data to produce an IDF idf_model
    ## This rescales hashed data by down-weighing features that occur frequently
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(featurized_data)
    rescaled_data = idf_model.transform(featurized_data)
    
    ## Normalize vectors
    ## Cosine similarity of L2 normalized vectors is just their dot products,
    ## therefore for ease of calculation we normalize the feature vectors extracted
    ## by both TF-idf and CV.
    normalizer=Normalizer(inputCol='features',outputCol='features_norm', p=2.0)
    normalized_data=normalizer.transform(rescaled_data)

    normalizer_cv=Normalizer(inputCol='cvfeatures', outputCol='cvfeatures_norm', p=2.0)
    normalized_data=normalizer_cv.transform(normalized_data)


    extracted_features_both=normalized_data.select('user_id','features_norm','cvfeatures_norm').rdd.mapPartitions(reformat_partition).cache()
    
    ## Create lookup table for both TF-idf and CV features.
    ## Lookup table consists of the original user_id and the extracted features
    vector_lookup_broadcast=sc.broadcast(extracted_features_both.collectAsMap())

    output_rdd=spark.sparkContext.parallelize([test_id]).cache()
    output_rdd.map(get_vector_info).mapValues(lambda x: (sorted(x[0], reverse=True)[:5], sorted(x[1], reverse=True)[:5])).values().saveAsTextFile('W1_out_.txt')


    ## Workload 2
    ## Extract data - user_id and user_mentions
    ## Explode such that each record in the DataFrame shows mapping between one user_id to one mentioned user_id
    ## Fault tolerance - filter our user_id null values.
    ## Extract just id value from mention_user_id column
    tweets_w2=tweets.select("user_id",explode("user_mentions")).filter("user_id IS NOT NULL").withColumn("col",f.col("col")["id"])
    pair_rating=tweets_w2.groupBy(tweets_w2.columns).count().select('user_id', f.hash('user_id').alias('hash_user_id'), f.col("col").alias('mention_user_id'), f.hash('col').alias('hash_mention_id'), f.col('count').alias('y')).cache()
    
    ## Broadcast  lookup RDDs, user_id -> hash and  mention_id -> hash
    id_hash_map=sc.broadcast(pair_rating.select("user_id","hash_user_id","mention_user_id", "hash_mention_id").rdd.mapPartitions(reformat_w2_partition).collect())

    ## Alternating Least Squares model
    ## User and Item ids inputted as their hash values due to ALS restrictions
    model=ALS(rank=10, seed=0, maxIter=5, regParam=0.1, implicitPrefs=True, alpha=1.0, userCol='hash_user_id', itemCol='hash_mention_id', ratingCol='y').fit(pair_rating)
    
    ## The user subset is all distinct users, can be collected by getting all distinct hash_user_ids
    user_subset=pair_rating.select("hash_user_id").distinct()
    top_predictions=model.recommendForUserSubset(user_subset, 5)

    top_predictions_rdd=top_predictions.rdd.mapValues(reformat_values_w2).map(convert_hash)
    top_predictions_rdd.saveAsTextFile('W2_out_.txt')
