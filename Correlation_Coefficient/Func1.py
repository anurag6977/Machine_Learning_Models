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
