import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, ChiSqSelector
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier, 
                                       RandomForestClassifier, NaiveBayes, LinearSVC, OneVsRest)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from xgboost import XGBClassifier
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("777 Project").getOrCreate()

# Load the dataset (replace with your actual path)
file_path = "C:\\Users\\avina\\Documents\\GitHub\\cs777-final-project-avinashnelluri\\diabetes_health_indicators.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Rename the label column
df = df.withColumnRenamed("Diabetes_012", "label")

# Initialize evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

def train_and_evaluate(model, train_data, test_data, model_name):
    """Train a model and evaluate its performance."""
    # Create a Pipeline
    pipeline = Pipeline(stages=[assembler, selector, model])
    
    # Train the model
    model_fitted = pipeline.fit(train_data)
    
    # Make predictions
    predictions = model_fitted.transform(test_data)
    
    # Evaluate the model
    accuracy = accuracy_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    
    # Print results
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} F1-Score: {f1_score:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    
    # Show confusion matrix
    confusion_matrix = predictions.groupBy("label", "prediction").count().toPandas()
    #plot_confusion_matrix(confusion_matrix, model_name)

def plot_confusion_matrix(confusion_matrix_df, model_name):
    """Plot the confusion matrix."""
    confusion_matrix_df = confusion_matrix_df.pivot(index='label', columns='prediction', values='count').fillna(0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_df, annot=True, fmt=".0f", cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Define split ratios
split_ratios = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]

# Loop over different train-test splits
for split_ratio in split_ratios:
    print(f"\nRunning models for train-test split: {split_ratio[0]*100:.0f}% train, {split_ratio[1]*100:.0f}% test\n")
    
    # Split the dataset into training and test sets
    train_df, test_df = df.randomSplit(split_ratio, seed=42)
    feature_columns = df.columns[1:]
    
    # Create a VectorAssembler to combine feature columns into a single vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    
    # Feature selection using Chi-Squared selector
    selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    
    # Handle class imbalance by oversampling minority classes
    # Calculate the maximum class count
    class_distribution = train_df.groupBy("label").count().collect()
    max_class_count = max([row['count'] for row in class_distribution])

    # Oversample each minority class
    oversampled_train_df = train_df
    for row in class_distribution:
        label = row['label']
        count = row['count']
        if count < max_class_count:
            # Calculate how many times to duplicate the data
            fraction = (max_class_count / count) - 1  # How much to oversample
            additional_df = train_df.filter(F.col("label") == label).sample(withReplacement=True, fraction=fraction)
            
            # Append the additional rows
            oversampled_train_df = oversampled_train_df.unionAll(additional_df)

    # Optional: Print class distribution after oversampling
    oversampled_train_df.groupBy("label").count().show()

    # Train and evaluate Logistic Regression
    lr = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.8, family="multinomial", 
                            featuresCol="selectedFeatures", labelCol="label")
    train_and_evaluate(lr, oversampled_train_df, test_df, "Logistic Regression")
    
    # Train and evaluate Decision Tree
    dt = DecisionTreeClassifier(featuresCol="selectedFeatures", labelCol="label")
    train_and_evaluate(dt, oversampled_train_df, test_df, "Decision Tree")
    
    # Train and evaluate Random Forest
    rf = RandomForestClassifier(featuresCol="selectedFeatures", labelCol="label", numTrees=100, maxDepth=10)
    train_and_evaluate(rf, oversampled_train_df, test_df, "Random Forest")
    
    # Train and evaluate Naive Bayes
    nb = NaiveBayes(featuresCol="selectedFeatures", labelCol="label", modelType="multinomial")
    train_and_evaluate(nb, oversampled_train_df, test_df, "Naive Bayes")
    
    # Train and evaluate Linear SVC with OneVsRest
    lsvc = LinearSVC(featuresCol="selectedFeatures", labelCol="label", maxIter=100, regParam=0.1)
    ovr = OneVsRest(classifier=lsvc)
    train_and_evaluate(ovr, oversampled_train_df, test_df, "SVM")
    
    # Train and evaluate XGBoost Classifier
    train_pd = oversampled_train_df.toPandas()
    test_pd = test_df.toPandas()
    
    # Define and train the XGBoost Classifier
    xgb = XGBClassifier(n_estimators=100, max_depth=10, objective='multi:softprob', num_class=3)
    xgb.fit(train_pd[feature_columns], train_pd['label'])
    
    # Make predictions on the test set
    predictions_xgb = xgb.predict(test_pd[feature_columns])
    test_pd['prediction'] = predictions_xgb.astype(float)  # Ensure predictions are of type float
    
    # Convert predictions back to Spark DataFrame
    predictions_df_xgb = spark.createDataFrame(test_pd)
    
    # Cast prediction column to DoubleType
    predictions_df_xgb = predictions_df_xgb.withColumn("prediction", predictions_df_xgb["prediction"].cast("double"))
    
    # Evaluate XGBoost predictions using the predefined evaluators
    accuracy_xgb = accuracy_evaluator.evaluate(predictions_df_xgb)
    f1_score_xgb = f1_evaluator.evaluate(predictions_df_xgb)
    precision_xgb = precision_evaluator.evaluate(predictions_df_xgb)
    recall_xgb = recall_evaluator.evaluate(predictions_df_xgb)
    
    # Print results for XGBoost
    print(f"XGBClassifier Accuracy: {accuracy_xgb:.4f}")
    print(f"XGBClassifier F1-Score: {f1_score_xgb:.4f}")
    print(f"XGBClassifier Precision: {precision_xgb:.4f}")
    print(f"XGBClassifier Recall: {recall_xgb:.4f}")
    
    # Show confusion matrix for XGBoost
    confusion_matrix_xgb = predictions_df_xgb.groupBy("label", "prediction").count().toPandas()
    #plot_confusion_matrix(confusion_matrix_xgb, "XGBoost Classifier")