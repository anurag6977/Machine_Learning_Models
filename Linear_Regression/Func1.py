import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# correlation coefficient
class correlation_coefficient:
    def Get(self):
        self.meal_amount=np.array([34,108,64,88,99,51])
        self.tip_amount=np.array([5,17,11,8,14,5])

    def Show(self):
        self.Sum_multiply_xy=np.sum(self.meal_amount*self.tip_amount)
        self.sum_x=np.sum(self.meal_amount)
        self.sum_y=np.sum(self.tip_amount)
        self.n=len(self.meal_amount)
        self.squar_sum_x=np.sum(self.meal_amount*self.meal_amount)
        self.squar_sum_y=np.sum(self.tip_amount*self.tip_amount)

        self.R=((self.n*self.Sum_multiply_xy)-(self.sum_x*self.sum_y))

        self.cx=(self.n*self.squar_sum_x)-(self.sum_x*self.sum_x)
        self.cy=(self.n*self.squar_sum_y)-(self.sum_y*self.sum_y)

        self.D=math.sqrt(self.cx*self.cy)

        self.cr=self.R/self.D
        print('correlation coefficient :',f'{self.cr:.2f}')

        plt.scatter(self.meal_amount, self.tip_amount, color='blue', label='Data points')
        
        coeffs = np.polyfit(self.meal_amount, self.tip_amount, 1)  # Linear regression
        regression_line = np.poly1d(coeffs)
        
        plt.plot(self.meal_amount, regression_line(self.meal_amount), color='red', label='Regression Line')
        
        plt.xlabel('Meal Amount')
        plt.ylabel('Tip Amount')
        plt.title('Meal Amount vs Tip Amount')
        plt.legend()
        
        plt.grid(True)
        plt.show()

# Linear Regression
class Linear_Regression:
    def Get(self):
        self.bill_amount=int(input("Enter Bill Amount :"))
        self.meal_amount=np.array([34,108,64,88,99,51])
        self.tip_amount=np.array([5,17,11,8,14,5])
    def Show(self):
        self.mean_tb=np.mean(self.meal_amount)
        self.mean_ta=np.mean(self.tip_amount)

        self.bd=np.array(self.meal_amount-self.mean_tb)
        self.td=np.array(self.tip_amount-self.mean_ta)

        self.dp=self.bd*self.td

        self.sqar_bds=self.bd*self.bd

        self.mean_dp=np.sum(self.dp)

        self.mean_sqar_bds=np.sum(self.sqar_bds)

        self.b1=self.mean_dp/self.mean_sqar_bds

        self.b0=self.mean_ta-self.b1*self.mean_tb

        self.predicted_tip=(self.b0+self.b1*self.bill_amount)
        print(self.predicted_tip)

        plt.scatter(self.meal_amount, self.tip_amount, color='blue', label='Data Points')
        
        regression_line = self.b0 + self.b1 * self.meal_amount
        plt.plot(self.meal_amount, regression_line, color='red', label='Regression Line')
        
        plt.scatter(self.bill_amount, self.predicted_tip, color='green', label='Prediction', zorder=5)
        
        plt.xlabel('Meal Amount')
        plt.ylabel('Tip Amount')
        plt.title('Linear Regression: Meal Amount vs Tip Amount')
        plt.legend()
        
        plt.grid(True)
        plt.show()

# Prediction Error
class Prediction_error:
    def Get(self):
        self.meal_amount=np.array([34,108,64,88,99,51])
        self.tip_amount=np.array([5,17,11,8,14,5])
    def Show(self):
        self.mean_tb=np.mean(self.meal_amount)
        self.mean_ta=np.mean(self.tip_amount)

        self.bd=np.array(self.meal_amount-self.mean_tb)
        self.td=np.array(self.tip_amount-self.mean_ta)

        self.dp=self.bd*self.td

        self.sqar_bds=self.bd*self.bd

        self.mean_dp=np.sum(self.dp)

        self.mean_sqar_bds=np.sum(self.sqar_bds)

        self.b1=self.mean_dp/self.mean_sqar_bds
        self.b0=self.mean_ta-self.b1*self.mean_tb


        self.predict_tip=(self.b0+self.b1*self.meal_amount)
        self.diff=self.tip_amount-self.predict_tip
        print(self.diff)

        plt.scatter(self.meal_amount, self.tip_amount, color='blue', label='Actual Data')

        regression_line = self.b0 + self.b1 * self.meal_amount
        plt.plot(self.meal_amount, regression_line, color='red', label='Regression Line')

        for i in range(len(self.meal_amount)):
            plt.plot([self.meal_amount[i], self.meal_amount[i]],
                     [self.tip_amount[i], self.predict_tip[i]],
                     color='green', linestyle='dotted')

        plt.xlabel('Meal Amount')
        plt.ylabel('Tip Amount')
        plt.title('Linear Regression: Meal Amount vs Tip Amount')
        plt.legend()

        plt.grid(True)
        plt.show()


# Logistic Regression
class Logistics_Regression:
    def Get(self):
        self.x1=int(input("Enter your credit :"))
        self.credit_score=np.array([655, 692,  681, 663, 688, 693, 699, 699, 683, 698, 655, 703, 704, 745, 702])
        self.approval=np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])

    def Show(self):
        self.mean_credit_score=np.mean(self.credit_score)
        self.mean_approval=np.mean(self.approval)

        # Differences from mean
        self.bd=np.array(self.credit_score-self.mean_credit_score)
        self.td=np.array(self.approval-self.mean_approval)

        # Dot products  
        self.dp=self.bd*self.td

        # squares
        self.sqar_bds=self.bd*self.bd

        # sum of mean value 
        self.mean_dp=np.sum(self.dp)
        self.mean_sqar_bds=np.sum(self.sqar_bds)

        self.b1=self.mean_dp/self.mean_sqar_bds # Slope

        self.b0=self.mean_approval-self.b1*self.mean_credit_score # Intercept

        # Prediction for a new value
        p=math.exp(self.b0+self.b1*self.x1)/(1+math.exp(self.b0+self.b1*self.x1))
        print("predicted_probability :",f'{p:.0f}')

        print("Approved") if 1>=p else print("Not Approved")

        # Generating a range of credit scores for the plot
        credit_range = np.linspace(min(self.credit_score), max(self.credit_score), 100)
        probabilities = [math.exp(self.b0 + self.b1 * x) / (1 + math.exp(self.b0 + self.b1 * x)) for x in credit_range]

        # Scatter plot of actual data
        plt.scatter(self.credit_score, self.approval, color='blue', label='Actual Data')

        # Logistic regression curve
        plt.plot(credit_range, probabilities, color='red', label='Logistic Regression Curve')

        # Labels and title
        plt.xlabel('Credit Score')
        plt.ylabel('Approval Probability')
        plt.title('Logistic Regression: Credit Score vs Approval Probability')
        plt.legend()

        # Display plot
        plt.grid(True)
        plt.show()
            
# ID3 Algorithm
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
class ID3:
    def Get(self,data):
        self.data=data
        self.Outlook=input("Enter Outlook :")
        self.Temp=input("Enter Temp :")
        self.Humidity=input("Enter Humidity :")
        self.Wind=input("Enter Wind :")

        self.encoder = LabelEncoder()

        for column in self.data.columns:
            self.data[column] = self.encoder.fit_transform(self.data[column])

        self.X = data.drop('PlayTennis', axis=1)
        self.y = data['PlayTennis']

        self.model = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.model.fit(self.X, self.y)

        self.sample_input = pd.DataFrame([{
            'Outlook': f'{self.Outlook}', 
            'Temp': f'{self.Temp}', 
            'Humidity': f'{self.Humidity}', 
            'Wind': f'{self.Wind}'
            }])

        for column in self.sample_input.columns:
            self.sample_input[column] = self.encoder.fit_transform(self.sample_input[column])
        

    def Show(self):
        self.prediction = self.model.predict(self.sample_input)
        print(f"Prediction: {'Play Tennis' if self.prediction[0] == 1 else 'Do not Play Tennis'}")

        # Visualize the Decision Tree
        plt.figure(figsize=(12, 8))
        plot_tree(self.model, feature_names=self.X.columns, class_names=['Do not Play Tennis', 'Play Tennis'], filled=True, rounded=True)
        plt.title("Decision Tree Visualization")
        plt.show()

# KMeans Clustering
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters
        self.kmeans = None
        self.labels = None
        self.centroids = None

    def fit(self):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(self.data)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

    def show_results(self):
        print("Final Assignment")
        print("{:<10} {:<5} {:<5} {:<10}".format("Dataset No", "X", "Y", "Assignment"))
        for i, (x, y) in enumerate(self.data):
            print("{:<10} {:<5} {:<5} {:<10}".format(i + 1, x, y, self.labels[i] + 1))  # Adding 1 to labels for 1-based indexing

        print("\nCluster Centroids:")
        for i, centroid in enumerate(self.centroids):
            print(f"Cluster {i + 1}: {centroid}")

        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i in range(self.n_clusters):
            cluster_points = self.data[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i], label=f'Cluster {i + 1}')

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=200, color='yellow', edgecolors='black', marker='X', label='Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()


    

