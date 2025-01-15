import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ID3 Algorithm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

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

