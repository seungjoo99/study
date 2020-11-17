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


def sigmoid(x): #들어오는 x 값을 sigmoid 함수에 넣은 값으로 return
    x = np.clip(x, -100, None) #overflow를 방지하기위해 x값을 -100이상 범위로 제한
    return 1 / (1 + np.exp(-x))


def feedforward(x, w1, w2): #w1은 input->hidden layer의 weight, w2는 hidden->output layer의 weight
    hidden_input = np.dot(x, w1) #input x의 특성 하나씩과 그에 해당하는 가중치들을 곱한 것을 다 더하기 위해 행렬곱사용
    hidden_output = sigmoid(hidden_input) #hidden layer node로 들어온 값을 sigmoid 함수에 대입
    hidden_output = np.array(hidden_output, ndmin=2) #w2와 행렬 곱을 위해 array를 2차원으로 바꿈
    out_input = np.dot(hidden_output, w2) #sigmoid를 거친 hidden layer의 node의 값들과 그에 해당하는 가중치들을 곱한 것을 다 더해주기 위해 행렬곱 사용
    return hidden_output, out_input


def softmax(x): #입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화, 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수
    x = np.clip(x, -100, None)  #overflow를 방지하기위해 x값을 -100이상 범위로 제한
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x).reshape(-1, 1) #데이터의 label과 같은 array shape을 위해 열을 1개로 만들어줌


def getCost(label, output):
    err = -(label - output) #각 input data가 가지고 있는 y값과 인공신경망으로 계산한 값의 차이
    return err


def one_hot(y): #y값에 해당하는 값의 번째 수만 1, 나머지 6개는 0인 array로 변환해주는 함수
    num = np.unique(y, axis=0) #y가 가지고 있는 값들 (1~7)
    num = num.shape[0] #y가 가지고 있는 값들의 개수 (7개)
    encode = np.eye(num)[y - 1] #인덱스는 0부터 시작하므로 y가 가진 값-1 index만 1, 나머지는 0인 array 만들기
    return encode


# function to compute gradients using back-propagation
def getGradient(x, err, w1, w2): #x는 input, err는 label과 계산한 output의 차이, w1은 input->hidden layer의 weight, w2는 hidden->output layer의 weight
    m = len(x) #x의 총 길이 (400개)
    hidden_output = feedforward(x, w1, w2)[0] #feedforward 함수가 hidden node의 output, output node의 input을 둘다 return 하므로 그 중 첫번째 값
    w2_grad = np.dot(hidden_output.T, err) / m #output layer의 가중치에 대한 gradient 계산
    # 뒤에서 training 시에 사용하는 cross-entropy를 사용하면 loss에 대한 weight의 편미분은 (계산값-label)에 hidden_output을 곱한 값을 모두 더한 평균 (레포트에 설명)

    err2hidden = np.dot(err, w2.T) * hidden_output * (1 - hidden_output) #hidden node의 sigmoid함수까지의 gradient
    w1_grad = np.dot(x.T, err2hidden) / m  #hidden layer의 가중치에 대한 gradient 계산

    return w1_grad, w2_grad

def predict(x, w1, w2): #output layer에 들어오는 값 중 가장 큰 값의 index 찾기
    out_input = feedforward(x, w1, w2)[1]
    out_input=softmax(out_input)
    return np.argmax(out_input, axis=1)

# function to compute accuracy of network
def get_accuracy(x, w1, w2, y): #x는 input, y는 label
    match = 0
    if predict(x, w1, w2) == np.argmax(y, axis=1):
        match += 1   #predict한 값과 one-hot encoding된 label에서 가장 큰 index와 같다면 match+1
    return match #같은 것이 몇 개인지 return

'''Main'''
# Load Data
np.random.seed(29)
file = io.loadmat('OlfactoryData.mat')
X_data = file['olf'][0, 0]['Signals']
idx = list(np.arange(42277))
idx = idx[:40000]
np.random.shuffle(idx)
X_data = X_data[idx, :]
y_data = file['olf'][0, 0]['SmellTypes']
y_data = y_data[idx]

# Construct network
# 1) parameters initialization
# 1-1) weights initialization
# #평균이 0, 표준편차가 1인 정규분포를 따르는 랜덤값들로 가중치 처음에 초기화
weights_in2hid = np.random.normal(0, 1, (400, 100))
#input data는 특성이 400개, hidden layer의 node는 100개로 설정
weights_hid2out = np.random.normal(0, 1, (100, 7))
#7개의 class로 분류하는 문제이므로 output layer의 node 개수는 7개

# 1-2) variable initialization
x_train = X_data[:36000] #train set으로 앞에서부터 36000개
y_train = y_data[:36000]
y_o_train = one_hot(y_data[:36000]) #y(label)을 one_hot encoding 해줌
x_test = X_data[36000:] #test set으로 뒤에서부터 4000개 (전체에서 train_set 제외한 만큼)
y_test = y_data[36000:]
y_o_test = one_hot(y_data[36000:]) #y(label)을 one_hot_encoding 해줌

tr_losses = [] #train_set의 loss 저장
tr_accuracy = [] #train_set의 정확도 저장
te_losses = [] #test_set의 loss 저장
te_accuracy = [] #test_set의 정확도 저장

# 1-3) hyper-parameters setting
alpha = 0.01 #learning rate 0.01로 설정
epoch = 200 #모든 데이터를 학습시키는 것을 200번 반복 --> 200이후로는
train_data_len = 36000
test_data_len = 4000

# 2) Training
for it in range(epoch): #epoch만큼 반복
    loss = 0
    match = 0
    for i in range(train_data_len): #모든 train_data 반복 (36000,400)개로 특성이 400개인 data가 36000개
        x = x_train[i] #x data 하나씩 불러오기
        x = np.array(x, ndmin=2) #행렬 계산을 위해 2차원으로 만들어주기
        y = y_o_train[i] #x data에 해당하는 label 값 불러오기
        m = len(x)
        out_input = feedforward(x, weights_in2hid, weights_hid2out)[1] #feedforward 통해 output node의 input 값 구하기
        a = softmax(out_input) #output node의 input 값은 activation function으로 softmax 사용
        err = getCost(y, a) #계산 값과 label 값의 차이
        w1_grad, w2_grad = getGradient(x, err, weights_in2hid, weights_hid2out)
        #차이를 통해 backpropagation을 위한 gradient 구하기
        # Weights update
        #구한 gradient에 learning rate 곱한 값을 원래 가중치에서 빼주면서 backpropagation (SGD) 실행
        weights_in2hid = weights_in2hid + (-alpha * w1_grad)
        weigths_hid2out = weights_hid2out + (-alpha * w2_grad)

        loss += np.sum(-y * np.log(a)) #반복문 돌면서 loss sum --> logistic regression(cross entropy 사용)
        match += get_accuracy(x, weights_in2hid, weights_hid2out, y) #match 개수 sum
    print("train:", it)
    print(loss / train_data_len)
    tr_losses.append((loss) / train_data_len) #loss의 평균을 1 epoch당 loss로 생각하여 저장
    print(match / train_data_len)
    tr_accuracy.append(match / train_data_len) #match의 평균을 1 epoch 당 accuracy로 생각하여 저장

    t_loss = 0
    t_match = 0

    for i in range(test_data_len): #batch 사용을 안해서 1 epoch이 1 iteration
        #1 epoch 마다 test set으로 loss, 정확도 측정 (train_set과 같은 원리로 weight update 존재 유무만 차이)
        x = x_test[i]
        x = np.array(x, ndmin=2)
        y = y_o_test[i]
        m = len(x)
        out_input = feedforward(x, weights_in2hid, weights_hid2out)[1]
        a = softmax(out_input)
        t_loss += np.sum(-y * np.log(a))
        t_match += get_accuracy(x, weights_in2hid, weights_hid2out, y)

    print("test:", it)
    print(t_loss / test_data_len)
    te_losses.append(t_loss / test_data_len)
    print(t_match / test_data_len)
    te_accuracy.append(t_match / test_data_len)

## Show results
# Calculate test accuracy (final accuracy)

print(te_accuracy[-1]) #학습이 종료된 직후의 test accurac

# Plot loss and accuracy curve
plt.subplot(2,1,1)
plt.title('Loss') #train, test set에 대한 각각 loss
plt.plot(tr_losses, label='train')
plt.plot(te_losses, label='test')
plt.legend()
plt.subplot(2,1,2)
plt.title('accuracy') #train, test set에 대한 각각 accuracy
plt.plot(tr_accuracy, label='train')
plt.plot(te_accuracy,label='test')
plt.legend()
plt.show()

