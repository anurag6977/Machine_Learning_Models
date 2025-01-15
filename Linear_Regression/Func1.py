import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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
