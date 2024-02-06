from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn import preprocessing
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import random

def tree_classify_copy():
    pandas_frame = pd.read_csv('train.csv')
    #Assuming that name has no impact on whether person survived or not
    #Is there even as way to turn names into features? IDK
    pandas_frame = pandas_frame.drop(columns=['Name'])
    labels = pandas_frame['Survived']
    #we can drop survived because we're already separated it into its own column
    inputs = pandas_frame.drop(columns=['Survived'])

    inputs = pre_process_data(inputs)

    X_copy = inputs.to_numpy()
    Y = labels.to_numpy()
    possible_num_components = list(range(1,19))
    possible_depth_values = list(range(1,30))
    possible_min_samples_split_vals = list(range(2,50))
    cur_best_max_depth = None
    cur_best_min_samples_split = None
    cur_best_num_components = None
    cur_best_average_accuracy = -1

    X = preprocessing.StandardScaler().fit_transform(X=X_copy)

    #for cur_num_components in possible_num_components:
        #pca = PCA(n_components=cur_num_components)
        #X = pca.fit_transform(X=X_copy)
    kFold=KFold(n_splits=10,shuffle=True)
    for cur_max_depth in possible_depth_values:
        for cur_min_sample_split in possible_min_samples_split_vals:
            #print(cur_num_components)
            print(cur_max_depth)
            print(cur_min_sample_split)
            
            cur_accuracies = []
            for train_index,test_index in kFold.split(X):
    
                X_train, X_valid, Y_train, Y_valid = X[train_index], X[test_index], Y[train_index], Y[test_index]
                #X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)

                clf = DecisionTreeClassifier(max_depth=cur_max_depth, min_samples_split=cur_min_sample_split, criterion='gini')
                clf = clf.fit(X_train, Y_train)

                #dot_data = tree.export_graphviz(clf, out_file=None) 
                #graph = graphviz.Source(dot_data) 
                #graph.render("Titantic_2")
                predictions = clf.predict(X_valid)
                correct = 0
                incorrect = 0
                for index, prediction in enumerate(predictions):
                    if prediction == Y_valid[index]:
                        correct += 1
                    else:
                        incorrect += 1
                
                cur_accuracy = float(correct/(correct+incorrect))
                cur_accuracies.append(cur_accuracy)
            if np.mean(np.array(cur_accuracies)) > cur_best_average_accuracy:
                cur_best_average_accuracy = np.mean(np.array(cur_accuracies))
                cur_best_max_depth = cur_max_depth
                cur_best_min_samples_split = cur_min_sample_split
                #cur_best_num_components = cur_num_components


    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    #pca = PCA(n_components=cur_best_num_components)
    #Fit to the training data NOT testing data
    #X = pca.fit_transform(X=X_copy)

    clf = DecisionTreeClassifier(max_depth=cur_best_max_depth, min_samples_split=cur_best_min_samples_split, criterion='gini')
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy()
    X[152][6] = 35.6271884892086
    #X = pca.transform(X=X) 
    X = preprocessing.StandardScaler().fit_transform(X=X)

    predictions = clf.predict(X)

    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_tree.csv", index=False)
    print(cur_best_average_accuracy)
    print(cur_best_num_components)
    print(cur_best_max_depth)
    print(cur_best_min_samples_split)



def nn_classify_copy():
    pandas_frame = pd.read_csv('train.csv')
    #Assuming that name has no impact on whether person survived or not
    #Is there even as way to turn names into features? IDK
    pandas_frame = pandas_frame.drop(columns=['Name'])
    labels = pandas_frame['Survived']
    #we can drop survived because we're already separated it into its own column
    inputs = pandas_frame.drop(columns=['Survived'])

    inputs = pre_process_data(inputs)

    X_copy = inputs.to_numpy()
    possible_num_components = list(range(1,19))
    #hopefully all of these are valid layer sizes
    possible_layer_sizes = list(range(1, 5))
    possible_batch_sizes =  list(range(20,200,20))
    cur_best_layer_size = None
    cur_best_batch_size = None
    cur_best_num_components = None
    cur_best_average_accuracy = -1
    #for cur_num_components in possible_num_components:
    #pca = PCA(n_components=cur_num_components)
    #X = pca.fit_transform(X=X_copy)
    X = preprocessing.StandardScaler().fit_transform(X=X_copy)
    Y = labels.to_numpy()


    kFold=KFold(n_splits=10, shuffle=True)
    for layer_size_1 in possible_layer_sizes:
        for layer_size_2 in possible_layer_sizes:
            for batch_size in possible_batch_sizes:
                #print(cur_num_components)
                print(layer_size_1)
                print(layer_size_2)
                print(batch_size)
                cur_accuracies = []
                for train_index,test_index in kFold.split(X):

                    X_train, X_valid, Y_train, Y_valid = X[train_index], X[test_index], Y[train_index], Y[test_index]

                    #4 has been the best layer size so far
                    #random_state=random.randint(0, 2**32-1)
                    clf = MLPClassifier(hidden_layer_sizes=np.asarray([layer_size_1, layer_size_2]), solver="adam", max_iter=1000, random_state=0, batch_size=batch_size,alpha=0.5)
                    clf = clf.fit(X_train, Y_train)

                    predictions = clf.predict(X_valid)
                    correct = 0
                    incorrect = 0
                    for index, prediction in enumerate(predictions):
                        if prediction == Y_valid[index]:
                            correct += 1
                        else:
                            incorrect += 1
                    cur_accuracy = float(correct/(correct+incorrect))
                    cur_accuracies.append(cur_accuracy)
                print(np.mean(np.array(cur_accuracies)))
                if np.mean(np.array(cur_accuracies)) > cur_best_average_accuracy:
                    cur_best_average_accuracy = np.mean(np.array(cur_accuracies))
                    #cur_best_num_components = cur_num_components
                    cur_best_layer_size_1 = layer_size_1
                    cur_best_layer_size_2 = layer_size_2
                    cur_best_batch_size = batch_size

    print("Got to here")
    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    #pca = PCA(n_components=cur_best_num_components)
    #Fit to the training data NOT testing data
    #X = pca.fit_transform(X=X_copy)
    clf = MLPClassifier(hidden_layer_sizes=np.asarray([cur_best_layer_size_1,cur_best_layer_size_2]), solver="adam", max_iter=1000, random_state=0, batch_size=cur_best_batch_size,alpha=0.5)
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy()
    #152 passenger in testing set doesn't have a far so just give them the average fare among others in the testing set
    X[152][6] = 35.6271884892086

    X = preprocessing.StandardScaler().fit_transform(X=X)

    #X = pca.transform(X=X)

    predictions = clf.predict(X)

    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_nn.csv", index=False)
    #print(cur_best_num_components)
    print(cur_best_layer_size_1)
    print(cur_best_layer_size_2)
    print(cur_best_batch_size)
    print(cur_best_average_accuracy)
    














#Method used to find SVM using one of the 4 possible kernels of linear, poly, rbf, or sigmoid, either ovo or ovr decision
#shape, and a regularization parameter, ranging from 1 to 10, which achieves highest k-fold cross validation accuracy
#on titantic training set
def kernel_classify():
    pandas_frame = pd.read_csv('train.csv')
    #Assuming that name has no impact on whether person survived or not
    #Is there even as way to turn names into features? IDK
    pandas_frame = pandas_frame.drop(columns=['Name'])
    labels = pandas_frame['Survived']
    #we can drop survived because we're already separated it into its own column
    inputs = pandas_frame.drop(columns=['Survived'])

    inputs = pre_process_data(inputs)

    X = inputs.to_numpy()
    Y = labels.to_numpy()


    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))
    clf = LinearSVC(dual=False, C=0.1)
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy()
    #152 passenger in testing set doesn't have a far so just give them the average fare among others in the testing set
    X[152][6] = 35.6271884892086 

    predictions = clf.predict(X)

    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_kernel.csv", index=False)


def nn_classify():
    pandas_frame = pd.read_csv('train.csv')
    #Assuming that name has no impact on whether person survived or not
    #Is there even as way to turn names into features? IDK
    pandas_frame = pandas_frame.drop(columns=['Name'])
    labels = pandas_frame['Survived']
    #we can drop survived because we're already separated it into its own column
    inputs = pandas_frame.drop(columns=['Survived'])

    inputs = pre_process_data(inputs)

    X_copy = inputs.to_numpy()

    #standard normalize inputs
    X = preprocessing.StandardScaler().fit_transform(X=X_copy)
    Y = labels.to_numpy()


    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    #Train MLP Classifier on training data using optimal hyperparameters
    clf = MLPClassifier(hidden_layer_sizes=np.asarray([3,4]), solver="adam", max_iter=1000, random_state=0, batch_size=120, alpha=0.3)
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy()
    #152 passenger in testing set doesn't have a far so just give them the average fare among others in the testing set
    X[152][6] = 35.6271884892086

    #standard normalize test data inputs
    X = preprocessing.StandardScaler().fit_transform(X=X)

    #make predictions on testing data
    predictions = clf.predict(X)

    #write predictions in correct csv format
    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_nn.csv", index=False)

def tree_classify():
    pandas_frame = pd.read_csv('train.csv')
    #Assuming that name has no impact on whether person survived or not
    #Is there even as way to turn names into features? IDK
    pandas_frame = pandas_frame.drop(columns=['Name'])
    labels = pandas_frame['Survived']
    #we can drop survived because we're already separated it into its own column
    inputs = pandas_frame.drop(columns=['Survived'])

    #preprocess data
    inputs = pre_process_data(inputs)

    #convert inputs and labels to numpy
    X_copy = inputs.to_numpy()
    Y = labels.to_numpy()

    #standard normalize inputs
    X = preprocessing.StandardScaler().fit_transform(X=X_copy)

    #preprocess the testing data
    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    #Train tree classifier to the training data using optimal hyperparameters
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=20)
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy()
    #152 passenger in testing set doesn't have a far so just give them the average fare among others in the testing set
    X[152][6] = 35.6271884892086

    #standard normalize test data inputs
    X = preprocessing.StandardScaler().fit_transform(X=X)

    #make predictions on testing data
    predictions = clf.predict(X)

    #write predictions in correct csv format
    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_tree.csv", index=False)

def pre_process_data(inputs):
    #Map female to 0 and male to 1 for the sex (this is essentially one hot encoding)
    inputs['Sex'] = inputs['Sex'].map({'female': 1, 'male': 0})

    #Do one hot encoding for the port of originn: If they originated from the respective port, put a 1 in the column, 0 otherwise
    one_hot = pd.get_dummies(inputs['Embarked'])
    one_hot['C'] = one_hot['C'].map({True: 1, False: 0})
    one_hot['Q'] = one_hot['Q'].map({True: 1, False: 0})
    one_hot['S'] = one_hot['S'].map({True: 1, False: 0})
    #Replace the embarked column with the one hot columns
    inputs = inputs.drop(columns=['Embarked'])
    inputs = inputs.assign(C=one_hot['C']).assign(Q=one_hot['Q']).assign(S=one_hot['S'])

    #Tickets seem pretty random, almost like random numbers, so drop the column
    inputs = inputs.drop(columns=['Ticket'])
    
    #For a lot of passengers in the training set, they have no listed age, so calculate the age of all the passengers that have an age
    #listed and set the age of the no age passengers to that average age
    no_age = 0
    has_age = 0
    total_age = 0
    for index, elem in enumerate(inputs['Age']):
        #This is true is the person's age is NaN
        if elem != elem:
            no_age += 1
        #This is true is the person's age is not NaN and it therefore exists
        #Add the person's age to the sum of all the ages so far
        else:
            has_age += 1
            total_age += elem
    #Simple average age calculation
    average_age = float(total_age/has_age)
    for index, elem in enumerate(inputs['Age']):
        #If an age is missing, just make the person's age the average age of the people that do have an age
        if elem != elem:
            inputs['Age'][index] = average_age

    #Now do one hot encoding for the cabin classes. I put CC instead of C because the C column is already used in the one hot encoding of port of origin
    columns_dict = {}
    possible_cabins = ['A', 'B', 'CC', 'D', 'E', 'F', 'G', 'T']
    #Getting lists set up
    for cabin in possible_cabins:
        columns_dict[cabin] = []

    #Iterate through all the cabins for all the passengers
    for elem in inputs['Cabin']:
        if elem == elem:
            #For some reason, some passengers have multiple classes (like F and G for example) in their cabin string, so if they have multiple
            #put 1's for all of them in the one hot encoding
            for cabin in possible_cabins:
                #Special case because we are not checking for CC in the cabin, we're checking for C instead
                if cabin == 'CC':
                    if 'C' in elem:
                        columns_dict['CC'].append(1)
                    else:
                        columns_dict['CC'].append(0)
                    continue
                if cabin in elem:
                    columns_dict[cabin].append(1)
                else:
                    columns_dict[cabin].append(0)
                    
        #No cabin for passenger, so just do all 0's for one hot encoding
        else:
            for cabin in possible_cabins:
                columns_dict[cabin].append(0)
    
    for cabin in possible_cabins:
        inputs[cabin] = columns_dict[cabin]

    #One hot encoding for cabin is done so drop that column
    inputs = inputs.drop(columns=['Cabin'])
    return inputs
    

def plot_data(x_data, y_data, file_name):
    plt.figure()
    plt.plot(x_data, y_data)
    plt.legend()
    plt.savefig(f"{file_name}.png")

if __name__ == "__main__":
    tree_classify()
    nn_classify()
    kernel_classify()