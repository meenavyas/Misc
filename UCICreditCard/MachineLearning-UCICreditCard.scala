import org.apache.spark.sql.types.StringType

import org.apache.spark.ml.classification.LogisticRegression

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param._

import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor

import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
sc.setLogLevel("OFF")

/////////////////////////////////////////////////////
val dforig = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("UCI_Credit_Card.csv")
dforig.show(1)
// ID is of type string rest of the types were detecetd properly
// Renamed column :default.payment.next.month" to "y"
val df1 = dforig.withColumn("ID", dforig.col("ID").cast(StringType)).withColumnRenamed("default.payment.next.month", "y");
/////////////////////////////////////////////////////
// weighting of records
val numNegatives = df1.filter(df1("y") === 0).count
val datasetSize = df1.count
val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize
val df = df1.withColumn("classWeightCol", when(df1("y") === 0.0, balancingRatio).otherwise(1.0 - balancingRatio))
df.printSchema()
df.show(1)
/////////////////////////////////////////////////////
// convert categorical variables using OneHotEncoder https://spark.apache.org/docs/latest/ml-features.html#onehotencoder:
// 1) SEX has integer values 1 or 2
// 2) MARRIAGE has integert values 0,1,2,3
// 3) AGE has integer values from 21 to 79 (assuming it is year)
// 4) EDUCATION has integer values 0 to 6
// 5) PAY_0,2,3,4,5,6 has values between -2 to 8
// Do we need to run indexer on SEX, MARRIAGE, AGE, EDUCTAION as they are already ok?

val categoricalCols = Array("SEX", "MARRIAGE", "AGE", "EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
// for each of these columns call indexer and encoder for example 
// val indexer = new StringIndexer().setInputCol("PAY_0").setOutputCol("PAY_0_Index")
// val encoder = new OneHotEncoder().setInputCol("PAY_0_Index").setOutputCol("Pay_0_Vec")
var transDF = df;
for(c <- categoricalCols) {
    val str1 = c+"_Index";
    val str2 = c+"_Vec";
    val indexer = new StringIndexer().setInputCol(c).setOutputCol(str1)
    val encoder = new OneHotEncoder().setInputCol(str1).setOutputCol(str2)
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    transDF = pipeline.fit(transDF).transform(transDF);
}
transDF.show(1)
// combine useful columns into a column named "features"
val assembler = new VectorAssembler().setInputCols(Array("LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "SEX_Vec", "MARRIAGE_Vec", "AGE_Vec", "EDUCATION_Vec", "PAY_0_Vec", "PAY_2_Vec", "PAY_3_Vec", "PAY_4_Vec", "PAY_5_Vec", "PAY_6_Vec")).setOutputCol("features")

val pipeline = new Pipeline().setStages(Array(assembler))
val dffull = pipeline.fit(transDF).transform(transDF)
dffull.show(1)
///////////////////////////////////////////////////
// Remove unnecessary columns
//val columns = Seq[String]("features", "y", "ID", "classWeightCol") 
//val columns = Seq[String]("features", "y", "ID")
//val colNames = columns.map(name => col(name))
//val dffull = midDF.select(colNames:_*)
///////////////////////////////////////////////////
// Split data into training (80%) and test (20%).
val splits = dffull.randomSplit(Array(0.8, 0.2), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
///////////////////////////////////////////////////
// Evaluators
// binary classification evaluator looks at "rawPrediction"
val binaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol("y").setRawPredictionCol("rawPrediction")
binaryClassificationEvaluator.setMetricName("areaUnderROC")
// regression evaluator looks at "prediction" not "rawPrediction"
val regressionEvaluator = new RegressionEvaluator().setLabelCol("y").setPredictionCol("prediction") 
regressionEvaluator.setMetricName("rmse")
//regressionEvaluator.setMetricName("mse")
//regressionEvaluator.setMetricName("r2")
//regressionEvaluator.setMetricName("mae")
///////////////////////////////////////////////////
// Logistic regression
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFeaturesCol("features").setLabelCol("y")  
val lrModel = lr.fit(training)
val predictions = lrModel.transform(training)
val areaTraining = binaryClassificationEvaluator.evaluate(predictions)
println("Area under ROC using Logistic Regression on training data = " + areaTraining)

val predictionsTest = lrModel.transform(test)
val areaTest = binaryClassificationEvaluator.evaluate(predictionsTest)
println("Area under ROC using Logistic Regression on test data = " + areaTest)
//////////////////////////////////////////
// Set Weight column on Logistic Regression
val lrBalanced = new LogisticRegression().setWeightCol("classWeightCol").setLabelCol("y").setFeaturesCol("features")
val lrBalancedModel = lrBalanced.fit(training)
val predictionsLrBalancedTest = lrBalancedModel.transform(test)
val areaLrBalancedTest = binaryClassificationEvaluator.evaluate(predictionsLrBalancedTest)
println("Area under ROC using Logisitic Regression with Weigh Column on test data = " + areaLrBalancedTest)

/////////////////////////////////////////////////////
// CV model 10 fold cross validation
val pipeline = new Pipeline().setStages(Array(lr))
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.05, 0.1, 0.2)).addGrid(lr.maxIter, Array(5, 10, 15)).build
val cv = new CrossValidator().setNumFolds(10).setEstimator(pipeline).setEstimatorParamMaps(paramGrid).setEvaluator(binaryClassificationEvaluator)
val cvModel = cv.fit(training)
val predictionsCvTest = cvModel.transform(test)
val areaCvTest = binaryClassificationEvaluator.evaluate(predictionsCvTest)
println("Area under ROC with Logistic Regression using Cross Validation on test data = " + areaCvTest)
/////////////////////////////////////////////////////
// Decision trees
val maxTreeDepth = 5 
val dt = new DecisionTreeRegressor().setLabelCol("y").setFeaturesCol("features").setMaxBins(32).setMaxDepth(5)
//setImpurity("Entropy") // or Gini
val pipelineDT = new Pipeline().setStages(Array(dt))
val modelDT = pipelineDT.fit(training) // train model 
val predictionsDTTest = modelDT.transform(test)
val rmseDT = regressionEvaluator.evaluate(predictionsDTTest)
println("Root Mean Squared Error (RMSE) Decision Trees on test data = " + rmseDT)
/////////////////////////////////////////////////////
// Random Forest (bagging)
val rf = new RandomForestRegressor().setLabelCol("y").setFeaturesCol("features")
val pipelineRF = new Pipeline().setStages(Array(rf))
val modelRF = pipelineRF.fit(training) // train model 
val predictionsRF = modelRF.transform(test) // Make predictions.
val rmseRF = regressionEvaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) Random Forest on test data = " + rmseRF)
// To print a tree 
//val rfModel = modelRF.stages(0).asInstanceOf[RandomForestRegressionModel]
// println("Learned regression forest model:\n" + rfModel.toDebugString)
/////////////////////////////////////////////////////
// Gradient Boosting
val gbt = new GBTRegressor().setLabelCol("y").setFeaturesCol("features").setMaxIter(10)
val pipelineGBT = new Pipeline().setStages(Array(gbt))
val modelGBT = pipelineGBT.fit(training)
val predictionsGBT = modelGBT.transform(test)
val rmseGB = regressionEvaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) Gradient Boosting on test data = " + rmseGB)
//val gbtModel = model.stages(0).asInstanceOf[GBTRegressionModel]
//println("Learned regression GBT model:\n" + gbtModel.toDebugString)
//////////////////////////////////////////
