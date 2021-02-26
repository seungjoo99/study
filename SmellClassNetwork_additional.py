'''
Introduction to Neural Engineering (Fall, 2020)
Implementation of Population Coding of Smell Sensing with Artificial Neural Network (ANN)
'''

'''
PLEASE FILL UP BELOW (PERSONAL INFO)
'''
# Full Name (last name first) : '''Park seung joo'''
# Student Number : '''2018250029'''


# import modules (can't be added)
from scipy import io
import numpy as np
import math
import matplotlib.pyplot as plt

''' Function definition (Any function can be added) '''

# fuction to return sigmoid value(node function)
def sigmoid(X):

    a = 1/(1+np.exp(-X))

    return a

# function to compute feedforward calculation
def feedforward(X, W1, W2, W3):
    '''reform this function'''
    Z1 = np.matmul(W1, X)
    a1 = sigmoid(Z1)
    Z2 = np.matmul(W2, a1)
    a2 = sigmoid(Z2)
    Z3= np.matmul(W3,a2)
    a3= sigmoid(Z3)

    return Z1, a1, Z2, a2, Z3, a3

# function to compute costs
def getCost(label, output):
    one_hot = np.zeros((num_class, 1))
    one_hot[label-1] = 1

    loss = np.sum((one_hot-output)**2)

    return loss

# function to compute gradients using back-propagation
def getGradient(data, w2, w3, a1, a2, a3, label):
    '''reform this function'''
    one_hot = np.zeros((num_class, 1))
    one_hot[label-1] = 1

    dw3 = -2 * np.matmul(((one_hot - a3) * a3 * (1 - a3)), a2.T)
    dw2 = -2 * np.matmul(a2 * (1 - a2) * np.matmul(w3.T, ((one_hot - a3) * a3 * (1 - a3))), a1.T)
    k1= np.matmul(np.transpose((one_hot-a3) * a3 * (1-a3)), w3) * np.transpose(a2 * (1-a2))
    dw1 = -2 * np.matmul(a1 * (1 - a1) * np.transpose(np.matmul(k1, w2)), data.T)
    return dw1, dw2, dw3

# function to compute accuracy of network
def get_accuracy(correct, data_len):
    accuracy = correct / data_len * 100

    return accuracy

def predict(output):
    predict_label = np.argmax(output)

    return predict_label


'''Main'''
# Load Data
file = io.loadmat('OlfactoryData.mat')
X_data = file['olf'][0, 0]['Signals']
idx = list(np.arange(42277))
idx = idx[:40000]
np.random.shuffle(idx)
X_data = X_data[idx, :]
X_data = (X_data-np.mean(X_data))/np.std(X_data) # pre-processing, normalization
y_data = file['olf'][0, 0]['SmellTypes']
y_data = y_data[idx]
print("Total X_dataset shape: ", X_data.shape) # 40000, 400
print("Total y_dataset shape: ", y_data.shape, '\n') # 40000, 1

# Split train and validation set
X_train = X_data[:int(X_data.shape[0]*0.9*0.8), :]
y_train = y_data[:int(X_data.shape[0]*0.9*0.8)]
print("Train X_dataset shape: ", X_train.shape) # 28800, 400
print("Train y_dataset shape: ", y_train.shape) # 28800, 1

X_val = X_data[int(X_data.shape[0]*0.9*0.8):int(X_data.shape[0]*0.9), :]
y_val = y_data[int(X_data.shape[0]*0.9*0.8):int(X_data.shape[0]*0.9)]
print("Validation X_dataset shape: ", X_val.shape) # 7200, 400
print("Validation y_dataset shape: ", y_val.shape, '\n') # 7200, 1

X_test = X_data[int(X_data.shape[0]*0.9):, :]
y_test = y_data[int(X_data.shape[0]*0.9):]
print("Test X_dataset shape: ", X_test.shape) # 4000, 400
print("Test y_dataset shape: ", y_test.shape, '\n') # 4000, 1


# Construct network
# 1) parameters initialization
# 1-1) variable initialization
num_input_features = X_data.shape[1] # 400
num_class = len(np.unique(y_data)) # 7
train_data_len = X_train.shape[0]
val_data_len = X_val.shape[0]
test_data_len = X_test.shape[0]
losses_tr = []
losses_val = []
accs_tr = []
accs_val = []

# 1-2) hyper-parameters setting
learning_rate = 0.01
epoch = 50
num_hidden1_node = 256
num_hidden2_node= 64

# 1-3) weights initialization
weights_in2hid1 = np.random.normal(0, 0.01, (num_hidden1_node, num_input_features))
weights_hid12hid2 = np.random.normal(0, 0.01, (num_hidden2_node, num_hidden1_node))
weights_hid22out = np.random.normal(0, 0.01, (num_class, num_hidden2_node))

'''Fill in this line'''

# 2) Learning
for it in range(epoch):
    print('Epoch: ', it)
    correct_tr = 0
    loss_tr = 0
    correct_val = 0
    loss_val = 0

    # Training
    for i in range(train_data_len): # Stochastic gradient descent (SGD)
        _, a1, _, a2, _, a3 = feedforward(X_train[i, :].reshape(-1, 1), weights_in2hid1, weights_hid12hid2, weights_hid22out) # Reform this line
        loss = getCost(y_train[i, 0], a3) # Reform this line
        dw1, dw2, dw3 = getGradient(X_train[i, :].reshape(-1, 1), weights_hid12hid2 ,weights_hid22out, a1, a2, a3, y_data[i, 0]) # Reform this line

        # Weights update
        weights_in2hid1 -= learning_rate * dw1
        weights_hid12hid2-= learning_rate * dw2
        weights_hid22out -= learning_rate * dw3
        '''Fill in this line'''
    # Get training accuracy
    for i in range(train_data_len):
        _, a1, _, a2, _, a3 = feedforward(X_train[i, :].reshape(-1, 1), weights_in2hid1, weights_hid12hid2, weights_hid22out) # Reform this line
        loss = getCost(y_train[i, 0], a3) # Reform this line
        prediction = predict(a3) # Reform this line
        loss_tr += loss
        if y_train[i, 0] - 1 == prediction:
            correct_tr += 1

    accuracy_tr = get_accuracy(correct_tr, train_data_len)
    loss_tr = loss_tr / train_data_len
    print("Train_loss: %.04f" % loss_tr)
    print("Train_acc: %.02f %% ( %d / %d ) " % (accuracy_tr, correct_tr, train_data_len ))


    # Validation
    for i in range(val_data_len):
        _, a1, _, a2, _, a3 = feedforward(X_val[i, :].reshape(-1, 1), weights_in2hid1, weights_hid12hid2, weights_hid22out) # Reform this line
        loss = getCost(y_val[i, 0], a3) # Reform this line
        prediction = predict(a3) # Reform this line
        loss_val += loss
        if y_val[i, 0] - 1 == prediction:
            correct_val += 1

    # Get validation accuracy
    accuracy_val = get_accuracy(correct_val, val_data_len)
    loss_val = loss_val / val_data_len
    print("Validation_loss: %.04f" % loss_val)
    print("Validation_acc: %.02f %% ( %d / %d ) " % (accuracy_val, correct_val, val_data_len))

    losses_tr.append(loss_tr)
    losses_val.append(loss_val)
    accs_tr.append(accuracy_tr)
    accs_val.append(accuracy_val)


# 3) Show results
# 3-1) Get test accuracy (final accuracy)
loss_test = 0
correct_test = 0
for i in range(test_data_len):
    _, a1, _, a2, _, a3 = feedforward(X_test[i, :].reshape(-1, 1), weights_in2hid1, weights_hid12hid2, weights_hid22out) # Reform this line
    loss = getCost(y_test[i, 0], a3) # Reform this line
    prediction = predict(a3) # Reform this line
    loss_test += loss
    if y_test[i, 0] - 1 == prediction:
        correct_test += 1

accuracy_test = get_accuracy(correct_test, test_data_len)
loss_test = loss_test / test_data_len
print("\nTest accuracy: %.02f %% ( %d / %d ) " % (accuracy_test, correct_test, test_data_len))

# 3-2) Plot loss and accuracy curve
plt.figure()
plt.plot(losses_tr, 'y', label='Train_loss')
plt.plot(losses_val, 'r', label='Validation_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train_loss', 'Validation_loss'])
plt.show()

plt.figure()
plt.plot(accs_tr, 'y', label='Train_accuracy')
plt.plot(accs_val, 'r', label='Validation_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train_acc', 'Validation_acc'])
plt.show()
