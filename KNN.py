import csv
import math
import operator
import random
import os

class CustomException(Exception):
    '''
    custom exception class for any type of exception.
    '''
    def __init__(self, m):
        self.message = m
    def __str__(self):
        return self.message

class Data:
    '''
    All the dataset related tasks are done by the functions of this class.
    '''
    def __init__(self,datapath):
        
        if type(datapath)!=type(''):
            raise CustomException('You didn\'t enter the path of the data in string format')
        if not datapath.endswith('.csv'):
            raise CustomException('datapath does not point to a csv file, only .csv format supported!')
        if not os.path.exists(datapath):
            raise CustomException('datapath does not exist!')
        
        self.datapath = datapath
        self.dataset = []
        self.X = []
        self.Y = []
        self.Y_categorical = []
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
    
    
    def load_dataset(self,separator='',is_label_string=False,columns_included=False):
        with open(self.datapath, 'r') as csvfile:
            lines = csv.reader(csvfile)
            dataset_raw = list(lines)
        dataset_raw = dataset_raw[1:] if columns_included else dataset_raw
        
        if ((type(dataset_raw[0])==type([])) and (len(dataset_raw[0])==1)):
            for i,row in enumerate(dataset_raw):
                dataset_raw[i] = row[0]
        
        if separator!='':
            if separator not in dataset_raw[0]:
                raise CustomException('Invalid Separator, not in data file!')
            
            float_range = len(dataset_raw[0].split(separator))-1 if is_label_string else len(dataset_raw[0].split(separator))
            for i,row in enumerate(dataset_raw):
                splited_row = row.split(separator)
                self.dataset.append(splited_row)
                for j in range(float_range):
                    self.dataset[i][j] = float(splited_row[j])
        
        else:
            self.dataset = dataset_raw
    
    def split_x_y(self):
        label_index = len(self.dataset[0])-1
        for i in range(len(self.dataset)):
            row = self.dataset[i]
            self.X.append(row[0:label_index])
            self.Y.append(row[label_index])
    
    def to_categorical(self):
        unique = set(self.Y)
        categorical_vector_length = len(unique)
        for i,label in enumerate(self.Y):
            temp = []
            for j,category in enumerate(unique):
                if label==category:
                    temp.append(1)
                else:
                    temp.append(0)
            self.Y_categorical.append(temp)
    
    def shuffle(self):
        indices = list(range(len(self.X)))
        random.shuffle(indices)
        
        X_temp = []
        Y_temp = []
        Y_cat_temp = []
        
        for index in indices:
            X_temp.append(self.X[index])
            Y_temp.append(self.Y[index])
            Y_cat_temp.append(self.Y_categorical[index])
        
        self.X = X_temp
        self.Y_categorical = Y_cat_temp
        self.Y = Y_temp
    
    def train_test_split(self,test_ratio=0.3,shuffle=False,categorical=False):
        if shuffle:
            self.shuffle()
        dataset_size = len(self.X)
        train_size = int((1-test_ratio)*dataset_size)
        self.X_train = self.X[0:train_size]
        self.X_test = self.X[train_size:]
        if categorical:
            self.Y_train = self.Y_categorical[0:train_size]
            self.Y_test = self.Y_categorical[train_size:]
        else:
            self.Y_train = self.Y[0:train_size]
            self.Y_test = self.Y[train_size:]
    
    def dataset_minmax(self,dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax
    
    def normalize_dataset(self):
        train_minmax = self.dataset_minmax(self.X_train)
        test_minmax = self.dataset_minmax(self.X_test)
        for row in self.X_train:
            for i in range(len(row)):
                row[i] = (row[i] - train_minmax[i][0]) / (train_minmax[i][1] - train_minmax[i][0])
        
        for row in self.X_test:
            for i in range(len(row)):
                row[i] = (row[i] - test_minmax[i][0]) / (test_minmax[i][1] - test_minmax[i][0])
                

def euclidean_distance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
    
    
def get_neighbors(x_train,y_train,test_instance, k):
    distances = []
    for x in range(len(x_train)):
        dist = euclidean_distance(test_instance, x_train[x])
        distances.append((x_train[x],dist,y_train[x]))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append((distances[x][0],distances[x][2]))
    return neighbors
    
def get_response(neighbors):
    classVotes = {}
    for x in range (len(neighbors)):
        response = neighbors[x][1]
        if response in classVotes.keys():
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse = True)
    return sortedVotes[0][0]
    
def get_accuracy(y_test, predictions):
    correct = 0
    for x in range(len(y_test)):
        if y_test[x] == predictions[x]:
            correct += 1
    return ( correct / float(len(y_test))) * 100.00
    
def main():
    data = Data('iris_dataset.csv')
    data.load_dataset(separator='\t',is_label_string=True,columns_included=True)
    data.split_x_y()
    data.to_categorical()
    data.train_test_split(test_ratio=0.3,shuffle=True)
    data.normalize_dataset()
    
    print('Train: ' + repr(len(data.X_train)))
    print('Test: ' + repr(len(data.X_test)))
    
    predictions = []
    k = 3    
    
    for x in range(len(data.X_test)):
        neighbors = get_neighbors(data.X_train, data.Y_train, data.X_test[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        # print('>predicted=',repr(result),', actual=',repr(testSet[x][-1]))
    accuracy = get_accuracy(data.Y_test, predictions)
    print('Accuracy = ',repr(accuracy),'%')
    
main()
