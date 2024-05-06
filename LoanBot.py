import os
import numpy as np
import pickle

class LoanBot:
    weights_file = "LoanBot.pkl"
    instance = None
    weights = None
    bias = None
    feature_mins = None
    feature_maxes = None

    def __init__(self):
        if self.instance != None:
            return self.instance
        
        if os.path.exists(self.weights_file):
            with open(self.weights_file, 'rb') as f:
                self.instance = pickle.load(f)
                self.weights = self.instance.weights
                self.bias = self.instance.bias
                self.feature_mins = self.instance.feature_mins
                self.feature_maxes = self.instance.feature_maxes
        else:
            self.weights = np.random.rand(4)
            self.bias = np.random.rand()
            self.instance = self
            self.save_instance()

    def save_instance(self):
        with open(self.weights_file, 'wb') as f:
            pickle.dump(self, f)

    def train(self, data):
        training_data = np.array(data)

        self.feature_mins = np.min(training_data[:, :-1], axis = 0)
        self.feature_maxes = np.max(training_data[:, :-1], axis = 0)

        for row in training_data:
            normalized_features = self.normalize_row(row[:-1])
            result = self.predict(normalized_features)
            if result != row[-1]:
                if row[4] == 1:
                    self.weights += normalized_features
                    self.bias += 1
                else:
                    self.weights -= normalized_features
                    self.bias -= 1

            self.save_instance()
        
    def test(self, data):
        testing_data = np.array(data)
        total_tests = 0
        total_correct = 0

        for row in testing_data:
            total_tests += 1
            #normalized_features = self.normalize_row(row[:-1])
            result = self.predict(row[:-1])
            print("Expected result: " + str(row[-1]))
            if result == row[-1]:
                total_correct += 1

        print(str((total_correct / total_tests) * 100) + "% accuracy")
        print("Tested " + str(total_tests) + " scenarios, and got " + str(total_correct) + " correct.")

    def predict(self, input_arr):
        weight_vector = np.array(self.weights)
        input_vector = self.normalize_row(np.array(input_arr))

        product = np.dot(weight_vector, input_vector)
        print("Product = " + str(product))
        print("Bias = " + str(self.bias))
        #product = np.clip(product, -100, 100)
        result = 1 / (1 + np.exp(-(product + (self.bias * 1))))
        print("Result = " + str(result))

        if result >= 1:
            return 1
        else:
            return 0
    
    def normalize_row(self, row):
        return (row - self.feature_mins) / (self.feature_maxes - self.feature_mins)
