# Evaluating the Impact of Combined Oversampling and Undersampling Techniques on Imbalanced Healthcare Data: A Case Study of Stroke Prediction

## Introduction
Imbalanced healthcare datasets pose significant challenges in building predictive models, particularly when the minority class represents critical outcomes such as stroke occurrences. Traditional machine learning models often fail to capture minority class instances, leading to biased results that compromise patient care.

This study evaluates the effectiveness of combining oversampling and undersampling techniques to address class imbalance in a stroke prediction dataset. The goal is to identify optimal resampling strategies that improve minority class detection while maintaining overall model performance.

## Dataset
The dataset used in this project contains information necessary to predict the occurrence of a stroke. Each row in the dataset represents a patient, and the dataset includes the following attributes:

- **id:** Unique identifier
- **gender:** "Male", "Female", or "Other"
- **age:** Age of the patient
- **hypertension:** 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- **heart_disease:** 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- **ever_married:** "No" or "Yes"
- **work_type:** "Children", "Govt_job", "Never_worked", "Private", or "Self-employed"
- **Residence_type:** "Rural" or "Urban"
- **avg_glucose_level:** Average glucose level in the blood
- **bmi:** Body mass index
- **smoking_status:** "Formerly smoked", "Never smoked", "Smokes", or "Unknown"
- **stroke:** 1 if the patient had a stroke, 0 if not

## Problem Definition
Class imbalance in healthcare datasets often leads to models with high accuracy but poor sensitivity for minority outcomes (e.g., stroke). This study focuses on:
1. Evaluating model performance on imbalanced data.
2. Applying sequential combinations of oversampling and undersampling techniques to balance the dataset.
3. Comparing performance metrics across resampling methods.
4. Recommending effective techniques for imbalanced healthcare datasets.

## Methods
1. **Data Preprocessing:** Cleaning and preparing the data for analysis.
2. **Baseline Model:** Training a baseline model on the dataset that applies **Oversampling Techniques** and **Undersampling Techniques**.
3. Sequential combination between **Oversampling Techniques** and **Undersampling Techniques**.
   
    3.1. **Oversampling Techniques:** 
   - **Random Over-Sampling:** Randomly replicating minority class examples to balance the dataset.
   - **SMOTE (Synthetic Minority Over-sampling Technique):** Generating synthetic examples for the minority class by interpolating between existing examples.
   - **ADASYN (Adaptive Synthetic Sampling):** Similar to SMOTE, but adaptively focuses on harder-to-learn examples by generating more synthetic data points in regions where the model struggles.
   - **Borderline-SMOTE:** An extension of SMOTE that focuses on generating synthetic data points near the borderline of the minority and majority classes.
   - **SVMSMOTE:** Uses support vector machines (SVM) to create synthetic instances that are close to the decision boundary.
   - **SMOTEENN:** A combination of SMOTE and Edited Nearest Neighbors (ENN) that first generates synthetic samples using SMOTE and then cleans the dataset using ENN.
   - **SMOTETomek:** Combines SMOTE and Tomek Links to generate synthetic samples and then remove noisy samples.
     
   3.2. **Undersampling Techniques:** 
   - **Cluster Centroids:** Reduces the dataset by synthesizing centroids from clusters using K-means, rather than selecting individual samples.
   - **Random Undersampling:** Randomly selects a subset of data from the targeted classes.
   - **NearMiss:** Uses three different rules based on nearest neighbors to select samples:
     - **NearMiss-1:** Chooses samples with the smallest average distance to the closest samples of the minority class.
     - **NearMiss-2:** Chooses samples with the smallest average distance to the farthest samples of the minority class.
     - **NearMiss-3:** Keeps samples based on a two-step nearest-neighbor selection process.
   - **Tomek Links:** Removes pairs of samples from different classes that are each other's closest neighbors.
   - **Edited Nearest Neighbors (ENN):** Removes majority samples if any or most neighbors belong to a different class.
   - **Repeated Edited Nearest Neighbors (RENN):** Extends ENN by repeating the process multiple times.
   - **All KNN:** Increases the neighborhood size in each iteration of ENN to progressively clean the dataset.
   - **Condensed Nearest Neighbor (CNN):**  Adds samples that are difficult to classify based on a 1-nearest neighbor rule.
   - **One-Sided Selection (OSS):** Combines CNN and Tomek Links to remove noisy or difficult-to-classify majority samples.
   - **Neighborhood Cleaning Rule (NCR):** Uses a combination of ENN and nearest-neighbor classification to remove noisy samples.
   - **Instance Hardness Threshold:**  Retains samples that are easier to classify by removing instances classified with low probability, based on a specified classifier.
   
     **Source:** https://imbalanced-learn.org/stable/under_sampling.html

   3.3. **Pairs of Sequential Combinations**
   
     3.3.1 **Random Over-Sampling**
     - Random Over-Sampling + Cluster Centroids
     - Random Over-Sampling + Random Undersampling
     - Random Over-Sampling + NearMiss-1
     - Random Over-Sampling + NearMiss-2
     - Random Over-Sampling + NearMiss-3
     - Random Over-Sampling + Tomek Links
     - Random Over-Sampling + ENN
     - Random Over-Sampling + RENN
     - Random Over-Sampling + All KNN
     - Random Over-Sampling + CNN
     - Random Over-Sampling + OSS
     - Random Over-Sampling + NCR
     - Random Over-Sampling + IHT
       
   3.3.2 **SMOTE**
   - SMOTE + Cluster Centroids
   - SMOTE + Random Undersampling
   - SMOTE + NearMiss-1
   - SMOTE + NearMiss-2
   - SMOTE + NearMiss-3
   - SMOTE + Tomek Links
   - SMOTE + ENN
   - SMOTE + RENN
   - SMOTE + All KNN
   - SMOTE + CNN
   - SMOTE + OSS
   - SMOTE + NCR
   - SMOTE + IHT
   
   3.3.3 **ADASYN**
   - ADASYN + Cluster Centroids
   - ADASYN + Random Undersampling
   - ADASYN + NearMiss-1
   - ADASYN + NearMiss-2
   - ADASYN + NearMiss-3
   - ADASYN + Tomek Links
   - ADASYN + ENN
   - ADASYN + RENN
   - ADASYN + All KNN
   - ADASYN + CNN
   - ADASYN + OSS
   - ADASYN + NCR
   - ADASYN + IHT

   3.3.4 **Borderline-SMOTE**
   - Borderline-SMOTE + Cluster Centroids
   - Borderline-SMOTE + Random Undersampling
   - Borderline-SMOTE + NearMiss-1
   - Borderline-SMOTE + NearMiss-2
   - Borderline-SMOTE + NearMiss-3
   - Borderline-SMOTE + Tomek Links
   - Borderline-SMOTE + ENN
   - Borderline-SMOTE + RENN
   - Borderline-SMOTE + All KNN
   - Borderline-SMOTE + CNN
   - Borderline-SMOTE + OSS
   - Borderline-SMOTE + NCR
   - Borderline-SMOTE + IHT
   
   3.3.5 **SVMSMOTE**
   - SVMSMOTE + Cluster Centroids
   - SVMSMOTE + Random Undersampling
   - SVMSMOTE + NearMiss-1
   - SVMSMOTE + NearMiss-2
   - SVMSMOTE + NearMiss-3
   - SVMSMOTE + Tomek Links
   - SVMSMOTE + ENN
   - SVMSMOTE + RENN
   - SVMSMOTE + All KNN
   - SVMSMOTE + CNN
   - SVMSMOTE + OSS
   - SVMSMOTE + NCR
   - SVMSMOTE + IHT
   
   3.3.6 **SMOTEENN**
   - SMOTEENN + Cluster Centroids
   - SMOTEENN + Random Undersampling
   - SMOTEENN + NearMiss-1
   - SMOTEENN + NearMiss-2
   - SMOTEENN + NearMiss-3
   - SMOTEENN + Tomek Links
   - SMOTEENN + ENN
   - SMOTEENN + RENN
   - SMOTEENN + All KNN
   - SMOTEENN + CNN
   - SMOTEENN + OSS
   - SMOTEENN + NCR
   - SMOTEENN + IHT
   
   3.3.7 **SMOTETomek**
   - SMOTETomek + Cluster Centroids
   - SMOTETomek + Random Undersampling
   - SMOTETomek + NearMiss-1
   - SMOTETomek + NearMiss-2
   - SMOTETomek + NearMiss-3
   - SMOTETomek + Tomek Links
   - SMOTETomek + ENN
   - SMOTETomek + RENN
   - SMOTETomek + All KNN
   - SMOTETomek + CNN
   - SMOTETomek + OSS
   - SMOTETomek + NCR
   - SMOTETomek + IHT
     
5. **Model Training:** The training phase focused on evaluating machine learning models on both the original imbalanced dataset and datasets balanced using combinations of oversampling and undersampling techniques. Each resampling strategy aimed to enhance the model's ability to detect minority class instances while maintaining overall performance.
6. **Evaluation:** Comparing model performance using accuracy, precision, recall, F1-score, and AUC-ROC.

## Models
The following machine learning models were used in this study:
1. **Logistic Regression:** A linear model for binary classification that estimates probabilities using the logistic function. It's commonly used for classification tasks due to its simplicity and interpretability.
2. **Random Forest:** An ensemble learning method that creates multiple decision trees using random subsets of data and features, then averages their predictions to improve accuracy and control overfitting.
3. **Support Vector Machine (SVM):** A classifier that finds the optimal hyperplane to separate classes, maximizing the margin between the nearest data points of each class (support vectors). It can also provide probability estimates for classification tasks.
4. **Gradient Boosting:** An ensemble technique that builds models sequentially, with each model correcting the errors of the previous one. It often yields high accuracy, especially in structured data tasks.
5. **AdaBoost:** Another ensemble method, AdaBoost combines multiple weak classifiers to form a strong classifier, giving more weight to instances that were incorrectly classified in previous iterations. This approach is particularly useful for focusing on harder-to-classify instances.
6. **XGBoost:** An optimized gradient boosting framework that is highly efficient and scalable, widely used in competitions and industry applications. It includes regularization to prevent overfitting, making it suitable for both classification and regression.
7. **LightGBM:** A gradient boosting framework developed by Microsoft, optimized for speed and efficiency. It uses a novel tree-building strategy called "leaf-wise" growth and is known for handling large datasets well.
8. **CatBoost:** A gradient boosting library developed by Yandex, designed to handle categorical features efficiently without extensive preprocessing. It performs well on structured data and is robust against overfitting.
9. **k-Nearest Neighbors (k-NN):** A non-parametric, instance-based learning algorithm that classifies instances based on the majority class of their k nearest neighbors. It’s simple and effective for smaller datasets but can be computationally expensive with larger data.
10. **Decision Tree:** A non-parametric model that splits data based on feature values to form a tree-like structure. It’s interpretable and useful for both classification and regression tasks, though prone to overfitting.
11. **Naive Bayes:** A probabilistic classifier based on Bayes' theorem, with the assumption that features are independent. It’s efficient and performs well on high-dimensional data, especially with text classification.
12. **Linear Discriminant Analysis (LDA):** A linear classifier that projects data onto a lower-dimensional space, maximizing class separability. LDA is suitable for classification when the classes are linearly separable.
13. **Quadratic Discriminant Analysis (QDA):** An extension of LDA that models quadratic decision boundaries, allowing it to capture non-linear relationships between features and class labels.
14. **Extra Trees (Extremely Randomized Trees):** An ensemble method similar to Random Forest, but with more randomization in the tree-building process. It often improves generalization and is faster than Random Forest due to the reduced complexity in splitting nodes.

## Evaluation Metrics
The following evaluation metrics were used in this study:
- **Accuracy:** The ratio of correctly predicted instances to the total instances. Measures the overall correctness of the model. However, it can be misleading in imbalanced datasets, as it might reflect high values even if the model fails to predict the minority class correctly.
- **Precision:** The ratio of correctly predicted positive instances to the total predicted positives. Indicates the accuracy of the positive predictions made by the model. High precision means that there are fewer false positives.
- **Recall (Sensitivity or True Positive Rate):** The ratio of correctly predicted positive instances to all actual positives. Measures the model's ability to identify all relevant instances. High recall means that there are fewer false negatives.
- **F1-Score:** The harmonic mean of precision and recall. Provides a single metric that balances both precision and recall. It is particularly useful when the class distribution is imbalanced.
- **ROC AUC Score:** The area under the Receiver Operating Characteristic (ROC) curve.
   - The ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity). The AUC score indicates how well the model distinguishes between the classes. A score of 1 indicates perfect discrimination, while a score of 0.5 indicates no discrimination (random guessing).

## Results
This section details the performance of various machine learning models on datasets processed using undersampling techniques, oversampling techniques, and their sequential combinations.

### Performance on Oversampling Techniques
The table below summarizes the performance of several machine learning models when trained on the imbalanced stroke dataset using oversampling techniques:

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

### Performance on Undersampling Techniques
This section will summarize the performance of models when trained on datasets processed with undersampling techniques. 

### Performance on Combined Oversampling and Undersampling Techniques
Sequential combinations of oversampling and undersampling techniques were applied to evaluate their combined impact on model performance. 
#### Random Over-Sampling + Cluster Centroids

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + Random Undersampling

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + NearMiss-1

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + NearMiss-2

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + NearMiss-3

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + Tomek Links

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + ENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + RENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + All KNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + CNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + OSS
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + NCR
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Random Over-Sampling + IHT
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |


#### SMOTE + Cluster Centroids
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + Random Undersampling
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + NearMiss-1
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + NearMiss-2
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + NearMiss-3
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + Tomek Links
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + ENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + RENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + All KNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + CNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + OSS
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + NCR
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SMOTE + IHT
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
   
#### ADASYN + Cluster Centroids
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + Random Undersampling
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + NearMiss-1
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + NearMiss-2
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + NearMiss-3
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + Tomek Links
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + ENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + RENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + All KNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + CNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + OSS
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + NCR
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### ADASYN + IHT
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + Cluster Centroids
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + Random Undersampling
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + NearMiss-1
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + NearMiss-2
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + NearMiss-3
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + Tomek Links
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + ENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + RENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + All KNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + CNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### Borderline-SMOTE + OSS
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### Borderline-SMOTE + NCR
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### Borderline-SMOTE + IHT
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

#### SVMSMOTE + Cluster Centroids
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + Random Undersampling
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + NearMiss-1
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + NearMiss-2
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + NearMiss-3
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + Tomek Links
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + ENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + RENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + All KNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + CNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + OSS
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + NCR
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SVMSMOTE + IHT
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |   
#### SMOTEENN + Cluster Centroids
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + Random Undersampling
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + NearMiss-1
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + NearMiss-2
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + NearMiss-3
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + Tomek Links
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + ENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + RENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + All KNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + CNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + OSS
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + NCR
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTEENN + IHT
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + Cluster Centroids
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + Random Undersampling
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + NearMiss-1
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + NearMiss-2
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + NearMiss-3
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + Tomek Links
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + ENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + RENN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + All KNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + CNN
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + OSS
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + NCR
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |
#### SMOTETomek + IHT
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          |          |               |               |            |            |              |              |               |
| Random Forest                |          |               |               |            |            |              |              |               |
| Support Vector Machine       |          |               |               |            |            |              |              |               |
| Gradient Boosting            |          |               |               |            |            |              |              |               |
| XGBoost                      |          |               |               |            |            |              |              |               |
| AdaBoost                     |          |               |               |            |            |              |              |               |
| LightGBM                     |          |               |               |            |            |              |              |               |
| CatBoost                     |          |               |               |            |            |              |              |               |
| K-Nearest Neighbors          |          |               |               |            |            |              |              |               |
| Decision Tree                |          |               |               |            |            |              |              |               |
| Naive Bayes                  |          |               |               |            |            |              |              |               |
| Linear Discriminant Analysis |          |               |               |            |            |              |              |               |
| Quadratic Discriminant Analysis |      |               |               |            |            |              |              |               |
| Extra Trees                  |          |               |               |            |            |              |              |               |

# Conclusion

# Discussion

# Future Work
