 Project Overview

This project focuses on predicting obesity levels using machine learning techniques based on lifestyle, demographic, and physical attributes. The objective is to build a robust classification model capable of accurately identifying an individual’s obesity category.

The potential impact of this work lies in supporting early health risk detection, enabling more informed decisions in healthcare, fitness, and preventive medicine.

 Data Selection and Preparation

The dataset includes variables such as:

Demographic information: Age, Gender
Physical attributes: Height, Weight
Lifestyle habits: Diet, physical activity, transportation

Data preprocessing:
Handling categorical variables through:
Binary encoding
Ordinal encoding
One-hot encoding
Creating a clean and consistent feature set
Splitting the data into training and testing sets
Applying feature scaling (StandardScaler) to ensure compatibility with distance-based models like KNN

 Feature Engineering and Selection

A key step in this project was the creation of a new feature:

BMI (Body Mass Index) = Weight / Height²

This feature captures a more meaningful relationship between height and weight and proved to be highly informative.

Additional analysis:
Correlation analysis (heatmap)
Multicollinearity analysis (VIF)
Permutation feature importance

Despite high multicollinearity between Weight, Height, and BMI, experiments showed that keeping all features improved model performance, highlighting the importance of preserving informative redundancy in non-linear models.

 Model Building and Evaluation

We implemented a K-Nearest Neighbors (KNN) classifier, chosen because:

It performs well on structured/tabular data
It captures non-linear relationships
It is highly interpretable in terms of distance-based reasoning
Evaluation metrics:
Accuracy (primary metric)
Classification Report (precision, recall, F1-score)
Confusion Matrix (error analysis)

⚙️ Hyperparameter Tuning and Model Optimization

🔹 GridSearchCV
Exhaustive search across predefined parameter combinations

Parameters tested:

n_neighbors
weights
metric
🔹 Optuna
Adaptive, intelligent search

Parameters explored:

n_neighbors
weights
p (distance function)

 Results Comparison
Method	n_neighbors	weights	metric / p	CV Score	Test Accuracy
GridSearchCV	4	distance	manhattan	0.9107	0.8971
Optuna	5	distance	p = 1	0.9083	0.8876

 Best Model

The GridSearchCV model achieved the best overall performance:

Best CV Score: 0.9107
Best Test Accuracy: 0.8971

Optuna produced a very similar configuration (p = 1 ≈ Manhattan), confirming the robustness of the solution, but with slightly lower performance.

 Key Findings and Insights
The optimal configuration consistently used:
Distance-based weighting
Manhattan distance (or equivalent p = 1)
Feature redundancy (Weight, Height, BMI) did not harm performance; instead, it improved the model by enriching the feature space
Removing features like BMI or Height reduced performance, confirming that feature interactions are critical in KNN
GridSearchCV and Optuna converged to similar solutions, validating the stability of the model

Final Conclusion

The project successfully developed a high-performing classification model, achieving close to 90% accuracy.

Key takeaways:
Careful feature engineering + proper preprocessing + tuning are more impactful than model complexity
Simple models like KNN can achieve strong performance when optimized correctly
