import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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
            
