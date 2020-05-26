# Databricks notebook source
# library
from pyspark.sql.functions import udf, when, col, first, last, avg, lit
from pyspark.sql.types import BooleanType, IntegerType, StringType, DateType, FloatType
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import time, datetime

# SparkSessjon 객체 생성 -> 모든 Spark 작업의 시작점
spark = SparkSession \
.builder \
.appName("Homework") \
.getOrCreate()



# Csv File Read
step_df = spark.read.load("/FileStore/tables/step_data.csv", format="csv", sep=",",  header = 'True', inferSchema='True')
weight_df = spark.read.load("/FileStore/tables/weight_data_float.csv", format="csv", sep=",",  header = 'True', inferSchema='True')
glucose_df = spark.read.load("/FileStore/tables/glucose_data_float.csv", format="csv", sep=",", header = 'True', inferSchema='True')




# data type check
step_df.printSchema()
weight_df.printSchema()
glucose_df.printSchema()



# data type 바꿔주는 함수 선언

# NA 값을 null로 변경하기
def cleanString(value):
  if(value == None) or ('null' in value):
    return None
  else:
    return value

# 유닉스시간을 Date Type으로 변경
def stringToDate(value):
  if value is None:
    return None
  else:
    return datetime.datetime.fromtimestamp(float(float(value)/float(1000)))




# 3개 dataframe의 attribute 형변환
cleanStringFunction = udf(cleanString, StringType())
cleanDateFunction = udf(stringToDate, DateType())

step_df = step_df \
    .withColumn(
      'day_time', cleanDateFunction('day_time'))\
    .withColumn(
      # null로 변경
      'age_group', cleanStringFunction('age_group')) \
    .withColumn(
      # null로 변경
      'gender', cleanStringFunction('gender'))

weight_df = weight_df \
    .withColumn(
      'measurement_time', cleanDateFunction('measurement_time'))\
    .withColumn(
      # null로 변경
      'gender', cleanStringFunction('gender'))

glucose_df = glucose_df \
    .withColumn(
      'measurement_time', cleanDateFunction('measurement_time'))\
    .withColumn(
      # null로 변경
      'gender', cleanStringFunction('gender'))




# Sanity Check
glucose_df = glucose_df.withColumn("glucose", when(col("glucose").isNull() , None).otherwise(col("glucose")))\
                         .withColumn("glucose", when(col("glucose") <= 0, None).otherwise(col("glucose")))\
                        .withColumn("glucose", when(col("glucose") <= 5, 5).otherwise(col("glucose")))\
                        .withColumn("glucose", when(col("glucose") >= 95, 95).otherwise(col("glucose")))



# 하루에 여러 번 측정이 가능한 데이터 하루치 평균으로 처리
weight_daily_df = weight_df.groupBy('uid','measurement_time')\
.mean('weight')\
.orderBy('uid')

glucose_daily_df = glucose_df.groupBy('uid','measurement_time')\
.mean('glucose')\
.orderBy('uid')


# 과제 1 
# method 1 - 테이블을 uid당 각 데이터(step_count, weight, glucose) 전체 평균이라고 생각
step_mean_df = step_df.groupBy('uid', 'age_group').mean('step_count')
weight_mean_df = weight_df.groupBy('uid').mean('weight')
glucose_mean_df = glucose_df.groupBy('uid').mean('glucose')

joined_mean_df = step_mean_df.join(weight_mean_df, (step_mean_df['uid'] == weight_mean_df['uid']),)\
       .join(glucose_mean_df, (step_mean_df['uid'] == glucose_mean_df['uid']),)\
        .orderBy(step_mean_df['uid'])

joined_mean_aligned_df = joined_mean_df\
  .select(step_mean_df['avg(step_count)'].alias('step_count'), step_mean_df['age_group'],\
          weight_mean_df['avg(weight)'].alias('weight'), glucose_mean_df['avg(glucose)'].alias('glucose'))




# method 2 - uid로 세 테이블 단순 조인이라고 생각 
# 하루치 평균 테이블로 치환 후, 세 테이블 조인
joined_df = step_df.join(weight_daily_df, (step_df['uid'] == weight_daily_df['uid']),)\
       .join(glucose_daily_df, (step_df['uid'] == glucose_daily_df['uid']),)

joined_aligned_df = joined_df\
  .select(step_df['step_count'], step_df['age_group'], weight_daily_df['avg(weight)'].alias('weight'), glucose_daily_df['avg(glucose)'].alias('glucose'))

# Homework 1-1 Output 
joined_aligned_df.show()


step_daily_df = step_df.select('uid','day_time','step_count')



# uid list 얻기 
uid_list_df = step_daily_df.select('uid').distinct().union(\
weight_daily_df.select('uid').distinct()).union(\
glucose_daily_df.select('uid').distinct()).distinct().orderBy('uid')



info_df = step_df.join(weight_df, (step_df['uid'] == weight_df['uid'])  ,'full_outer' )\
  .join(glucose_df, (step_df['uid'] == glucose_df['uid'])  ,'full_outer' )\
.orderBy(step_df['uid'])



info_df = info_df.select(step_df['uid'].alias('step_uid'),weight_df['uid'].alias('weight_uid'),glucose_df['uid'].alias('glucose_uid'),\
              step_df['country_cd'].alias('step_country_cd'), weight_df['country_cd'].alias('weight_country_cd'),glucose_df['country_cd'].alias('glucose_country_cd'),\
              step_df['gender'].alias('step_gender'), weight_df['gender'].alias('weight_gender'),glucose_df['gender'].alias('glucose_gender'),\
              step_df['age_group'].alias('step_age_group'), weight_df['age_group'].alias('weight_age_group'),glucose_df['age_group'].alias('glucose_age_group'))



# Imputation using 3 tables
# step table을 기준으로 missing data를 채워준다
info_imputing_df = info_df.withColumn("step_uid", when(col("step_uid").isNull(),  when(col("weight_uid").isNull(), col("glucose_uid") ).otherwise(col("weight_uid")) ).otherwise(col("step_uid")))\
.withColumn("step_gender", when(col("step_gender").isNull(),  when(col("weight_gender").isNull(), col("glucose_gender") ).otherwise(col("weight_gender")) ).otherwise(col("step_gender")))\
.withColumn("step_country_cd", when(col("step_country_cd").isNull(),  when(col("weight_country_cd").isNull(), col("glucose_country_cd") ).otherwise(col("weight_country_cd")) ).otherwise(col("step_country_cd")))\
.withColumn("step_age_group", when(col("step_age_group").isNull(),  when(col("weight_age_group").isNull(), col("glucose_age_group") ).otherwise(col("weight_age_group")) ).otherwise(col("step_age_group")))


info_imputed_df = info_imputing_df\
.select(info_imputing_df.step_uid.alias('uid'),\
        info_imputing_df.step_country_cd.alias('country_cd'),\
        info_imputing_df.step_gender.alias('gender'),\
        info_imputing_df.step_age_group.alias( 'age_group'))\
.orderBy('uid')

# uid list data와 country_cd, gender, age_group 조인
res1 = uid_list_df.join(info_imputed_df, uid_list_df['uid'] == info_imputed_df['uid'])\
.drop(info_imputed_df['uid'])

# Data Reshaping

# 이미 weight,glucose table의 데이터를 일일 평균 데이터로 치환했기에 avg말고 first를 사용했습니다.
step_reshaped_df = step_daily_df.groupBy('uid').pivot("day_time").agg(first(col("step_count")))
step_reshaped_df = step_reshaped_df.withColumnRenamed("uid", "uid_s").withColumnRenamed("2020-04-13", "2020-04-13_s")

weight_reshaped_df = weight_daily_df.groupBy('uid').pivot("measurement_time").agg(first(col("avg(weight)")))
weight_reshaped_df = weight_reshaped_df.withColumnRenamed("uid", "uid_w").withColumnRenamed("2020-04-12", "2020-04-12_w")\
       .withColumnRenamed("2020-04-13", "2020-04-13_w")

glucose_reshaped_df = glucose_daily_df.groupBy('uid').pivot("measurement_time").agg(first(col("avg(glucose)")*18))
glucose_reshaped_df = glucose_reshaped_df.withColumnRenamed("uid", "uid_g").withColumnRenamed("2020-04-12", "2020-04-12_g")\
       .withColumnRenamed("2020-04-13", "2020-04-13_g")

health_data_df = step_reshaped_df\
.join(weight_reshaped_df,step_reshaped_df['uid_s']==weight_reshaped_df['uid_w'],'outer')\
.join(glucose_reshaped_df,step_reshaped_df['uid_s']==glucose_reshaped_df['uid_g'],'outer')\
.orderBy(step_reshaped_df['uid_s'])\
.drop(*['uid_w','uid_g'])

res2 = res1.join(health_data_df, health_data_df['uid_s']== res1['uid'])\
.drop('uid_s')

# Homework 2
res2.show()


