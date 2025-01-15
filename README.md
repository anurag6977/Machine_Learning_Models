# Machine_Learning_Models
## Developed models of machine learning
## Project Overview
- Implemented Correlation Coefficient to analyze variable relationships and Linear Regression for tip prediction, visualizing results with NumPy and Matplotlib.
- Developed Prediction Error analysis to assess model accuracy and implemented Logistic Regression for customer approval prediction using Sigmoid function.
- Built an ID3 Decision Tree for predicting tennis play based on weather conditions and applied KMeans Clustering for segmenting data into distinct clusters, visualizing the outcomes with sklearn and Matplotlib.

## 1. Correlation Coefficient:
- Implemented a Correlation Coefficient calculation to analyze the relationship between two variables (meal amount and tip amount).
- Utilized NumPy for efficient mathematical operations and Matplotlib to visualize the data and regression line.

![correlation_coefficient](https://github.com/user-attachments/assets/6b11b6a1-b84c-4874-97d3-586c26564fa3)

## 2. Linear Regression:
- Built a Linear Regression model to predict tip amounts based on meal amounts.
- Applied mean normalization and computed slope and intercept to derive the regression line.
- Developed a prediction function to forecast tip amounts and visualized the model's performance using Matplotlib.

![Linear Regression](https://github.com/user-attachments/assets/88725a8d-4c9d-47f8-961a-a3658e9c0d3d)

## Prediction Error:
- Developed a model to compute Prediction Errors by comparing actual tip values with predicted values from the linear regression model.
- Visualized the error distribution using scatter plots and dotted lines to show residuals.

![Prediction Error](https://github.com/user-attachments/assets/105fc1b6-45c5-4c79-a199-0630a60a6df6)

## 3. Logistic Regression:
- Implemented Logistic Regression for predicting customer approval probability based on credit score.
- Applied Sigmoid function to calculate probabilities and made predictions for a new credit score value.
- Visualized the Logistic Regression curve and plotted actual vs predicted outcomes using Matplotlib.

![Logistic Regression](https://github.com/user-attachments/assets/51f0553a-b1b5-42b8-8dfd-8b77e0ebe98c)


## 4. ID3 (Decision Tree):
- Developed an ID3 decision tree model using sklearn's DecisionTreeClassifier to predict whether a person will play tennis based on weather conditions (e.g., Outlook, Temp, Humidity, Wind).
- Utilized LabelEncoder to transform categorical variables and visualized the decision tree using plot_tree for interpretability.

![ID3 (Decision Tree)](https://github.com/user-attachments/assets/cfac2db0-9be0-4da3-9c99-3614b8495ff5)

## 5. KMeans Clustering:
- Implemented KMeans Clustering to segment data into distinct clusters based on two features (X and Y).
- Applied the KMeans algorithm to identify optimal centroids and visualized the clusters and centroids using scatter plots.
- Analyzed final cluster assignments and centroids to understand the distribution and separation of data points.

![KMeans Clustering](https://github.com/user-attachments/assets/59ee0ae3-bb67-4f0d-bbec-fe77fd667438)
