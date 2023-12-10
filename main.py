from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import random

def tree_classify():
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

    #P = float(X.shape[1])
    #X = center(X)
    #D,V = np.linalg.eigh(1/P*np.dot(X.T,X))
    #Top eigenvectors are in the back
    #V = V[:,-7:]
    #X = np.dot(X, V)
    #X = normalize(X)
    possible_depth_values = list(range(1,30))
    possible_min_samples_split_vals = list(range(2,50))

    cur_best_max_depth = None
    cur_best_min_samples_split = None
    cur_best_average_accuracy = -1

    kFold=KFold(n_splits=10,random_state=10,shuffle=True)
    for cur_max_depth in possible_depth_values:
        for cur_min_sample_split in possible_min_samples_split_vals:
            
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
    

    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    clf = DecisionTreeClassifier(max_depth=cur_best_max_depth, min_samples_split=cur_best_min_samples_split, criterion='gini')
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy() 

    predictions = clf.predict(X)

    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_tree.csv", index=False)
    print(cur_best_average_accuracy)
    print(cur_best_max_depth)
    print(cur_best_min_samples_split)


def nn_classify():
    pandas_frame = pd.read_csv('train.csv')
    #Assuming that name has no impact on whether person survived or not
    #Is there even as way to turn names into features? IDK
    pandas_frame = pandas_frame.drop(columns=['Name'])
    labels = pandas_frame['Survived']
    #we can drop survived because we're already separated it into its own column
    inputs = pandas_frame.drop(columns=['Survived'])

    inputs = pre_process_data(inputs)

    X = inputs.to_numpy()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = labels.to_numpy()

    #hopefully all of these are valid layer sizes
    possible_layer_sizes = list(range(4, 9))
    possible_batch_sizes = list((range(50,200,4)))
    cur_best_layer_size = None
    cur_best_average_accuracy = -1
    cur_best_batch_size = None

    kFold=KFold(n_splits=13, shuffle=True)
    for layer_size in possible_layer_sizes:
        for batch_size in possible_batch_sizes:
            print(layer_size)
            print(batch_size)
            cur_accuracies = []
            for train_index,test_index in kFold.split(X):

                X_train, X_valid, Y_train, Y_valid = X[train_index], X[test_index], Y[train_index], Y[test_index]

                #4 has been the best layer size so far
                #random_state=random.randint(0, 2**32-1)
                clf = MLPClassifier(hidden_layer_sizes=np.asarray([layer_size]), solver="adam", max_iter=2000,alpha=0.1, random_state=0, batch_size=batch_size, learning_rate_init=0.001)
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
                cur_best_layer_size = layer_size
                cur_best_batch_size = batch_size

    print("Got to here")
    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    clf = MLPClassifier(hidden_layer_sizes=np.asarray([cur_best_layer_size]), solver="adam", max_iter=2000,alpha=0.1, random_state=0, batch_size=cur_best_batch_size, learning_rate_init=0.001)
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy()
    #152 passenger in testing set doesn't have a far so just give them the average fare among others in the testing set
    X[152][6] = 35.6271884892086
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    predictions = clf.predict(X)

    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_nn.csv", index=False)
    print(cur_best_layer_size)
    print(cur_best_batch_size)
    print(cur_best_average_accuracy)
    

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
    
    decision_function_shapes = ["ovo", "ovr"]
    kernels = ["linear", "poly", "rbf", "sigmoid"]

    possible_reg_params = list(range(1,11))

    cur_best_dfs = None
    cur_best_kernel = None
    cur_best_reg_param = None
    cur_best_average_accuracy = -1

    kFold=KFold(n_splits=6,random_state=10,shuffle=True)
    for dfs in decision_function_shapes:
        for kernel in kernels:
            for reg_param in possible_reg_params:
            
                cur_accuracies = []
                for train_index,test_index in kFold.split(X):
        
                    X_train, X_valid, Y_train, Y_valid = X[train_index], X[test_index], Y[train_index], Y[test_index]
                    #X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)

                    clf = SVC(C=reg_param, decision_function_shape=dfs, kernel=kernel)
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
                if np.mean(np.array(cur_accuracies)) > cur_best_average_accuracy:
                    cur_best_average_accuracy = np.mean(np.array(cur_accuracies))
                    cur_best_dfs = dfs
                    cur_best_kernel = kernel
                    cur_best_reg_param = reg_param
    print(cur_best_average_accuracy)
    print(cur_best_dfs)
    print(cur_best_kernel)
    print(cur_best_reg_param)
    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    clf = SVC(C=cur_best_reg_param, decision_function_shape=cur_best_dfs, kernel=cur_best_kernel)
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy()
    #Filling in the missing fare so there's no nan error
    X[152][6] = 35.6271884892086 

    predictions = clf.predict(X)

    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_kernel.csv", index=False)
    

def center(X):
	X_means = np.mean(X,axis=1)[:,np.newaxis]
	X_normalized = X - X_means
	
	return X_normalized

def normalize(x):
	x_stds = np.std(x,axis = 1)[:,np.newaxis]

	ind = np.argwhere(x_stds < 10**(-2))
	if len(ind) > 0:
		ind = [v[0] for v in ind]
		adjust = np.zeros((x_stds.shape))
		adjust[ind] = 1.0
		x_stds += adjust

	x = x/x_stds
	return x

def pre_process_data(inputs):
    #Map female to 0 and male to 1 for the sex
    inputs['Sex'] = inputs['Sex'].map({'female': 1, 'male': 0})

    one_hot = pd.get_dummies(inputs['Embarked'])
    one_hot['C'] = one_hot['C'].map({True: 1, False: 0})
    one_hot['Q'] = one_hot['Q'].map({True: 1, False: 0})
    one_hot['S'] = one_hot['S'].map({True: 1, False: 0})
    inputs = inputs.drop(columns=['Embarked'])
    inputs = inputs.assign(C=one_hot['C']).assign(Q=one_hot['Q']).assign(S=one_hot['S'])

    #Tickets seem pretty random from what I can see so drop it for now
    inputs = inputs.drop(columns=['Ticket'])
    #X = inputs.to_numpy()
    #clf = DecisionTreeClassifier()
    #clf = clf.fit(X, Y)
    
    no_age = 0
    has_age = 0
    total_age = 0
    for index, elem in enumerate(inputs['Age']):
        if elem != elem:
            no_age += 1
        else:
            has_age += 1
            total_age += elem
    average_age = float(total_age/has_age)
    for index, elem in enumerate(inputs['Age']):
        #If an age is missing, just make the person's age the average age of the people that do have an age
        if elem != elem:
            inputs['Age'][index] = average_age

    columns_dict = {}
    possible_cabins = ['A', 'B', 'CC', 'D', 'E', 'F', 'G', 'T']
    #Getting set up
    for cabin in possible_cabins:
        columns_dict[cabin] = []

    for elem in inputs['Cabin']:
        if elem == elem:
            #For some reason, some passengers have multiple classes in their cabin string, so if they have multiple
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

    inputs = inputs.drop(columns=['Cabin'])
    return inputs
    
def eigenvalues():
    
    pandas_frame = pd.read_csv('train.csv')
    #Assuming that name has no impact on whether person survived or not
    #Is there even as way to turn names into features? IDK
    pandas_frame = pandas_frame.drop(columns=['Name'])
    labels = pandas_frame['Survived']
    #we can drop survived because we're already separated it into its own column
    inputs = pandas_frame.drop(columns=['Survived'])

    inputs = pre_process_data(inputs)

    X = inputs.to_numpy()
    
    D,V = np.linalg.eigh(np.dot(X.T,X))
    plot_data(list(range(len(D))), np.flip(D), "eigenvalues")
    print(np.flip(D))
    print(V[:,:3].shape)

def plot_data(x_data, y_data, file_name):
    plt.figure()
    plt.plot(x_data, y_data)
    plt.legend()
    plt.savefig(f"{file_name}.png")

if __name__ == "__main__":
    #tree_classify()
    #nn_classify()
    kernel_classify()




def nn_copy():
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

    #hopefully all of these are valid layer sizes
    possible_layer_sizes = list(range(80,101))
    cur_best_layer_1 = None
    cur_best_layer_2 = None
    cur_best_layer_3 = None
    cur_best_average_accuracy = -1

    kFold=KFold(n_splits=10,random_state=10,shuffle=True)
    for layer_1_size in possible_layer_sizes:
        for layer_2_size in possible_layer_sizes:
            for layer_3_size in possible_layer_sizes:

                cur_accuracies = []
                for train_index,test_index in kFold.split(X):
        
                    X_train, X_valid, Y_train, Y_valid = X[train_index], X[test_index], Y[train_index], Y[test_index]

                    clf = MLPClassifier(hidden_layer_sizes=np.asarray([layer_1_size, layer_2_size, layer_3_size]), max_iter=200, early_stopping=True, random_state=10)
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
                print(clf.n_iter_)
                if np.mean(np.array(cur_accuracies)) > cur_best_average_accuracy:
                    cur_best_average_accuracy = np.mean(np.array(cur_accuracies))
                    cur_best_layer_1 = layer_1_size
                    cur_best_layer_2 = layer_2_size
                    cur_best_layer_3 = layer_3_size
                if cur_best_average_accuracy > 0.8:
                    break
            if cur_best_average_accuracy > 0.8:
                    break
        if cur_best_average_accuracy > 0.8:
                    break

    testing_frame = pre_process_data(pd.read_csv('test.csv').drop(columns=['Name']))

    clf = MLPClassifier(hidden_layer_sizes=np.asarray([cur_best_layer_1, cur_best_layer_2, cur_best_layer_3]))
    clf = clf.fit(X, Y)

    X = testing_frame.to_numpy() 

    predictions = clf.predict(X)

    passengers = pd.read_csv('test.csv')['PassengerId']
    predictions = pd.Series(predictions, name="Survived")
    final_frame = pd.concat([passengers, predictions], axis=1)
    final_frame.to_csv("predictions_tree.csv", index=False)
    print(cur_best_layer_1)
    print(cur_best_layer_2)
    print(cur_best_layer_3)
    print(cur_best_average_accuracy)
    