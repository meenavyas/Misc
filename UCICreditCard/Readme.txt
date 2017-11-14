UCI Credit Card Readme
~~~~~~~~~~~~~~~~~~~~~~

Refef my blog https://meenavyas.wordpress.com/2017/08/21/analysing-credit-card-default-datasets-using-apache-spark-and-scala/

Dataset is from https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
description of fields is in https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

I have modified "default.payment.next.month" to "y" in csv file.

Steps :
Download : spark-2.2.0-bin-hadoop2.7.tgz from https://spark.apache.org/downloads.html
cd spark-2.2.0-bin-hadoop2.7
Put these files there

./bin/spark-shell
:load MachineLearning-UCICreditCard.scala.scala

We can tune GBM parameters and make it better.
