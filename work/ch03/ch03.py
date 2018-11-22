#%% [markdown]
# 3.3.4 ニューラルネットワークの行列の積
#%%
import numpy as np
X = np.array([1, 2])
X.shape 
#%%
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
W.shape
#%%
Y = np.dot(X, W)
print(Y)
#%%
# 3層ニューラルネットワークの実装

#%%
def sigmoid(x):
    return 1/1+np.exp(-x)

def identify_function(x):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5]
                             ,[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4]
                             ,[0.2, 0.5]
                             ,[0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])                          
    network['W3'] = np.array([[0.1, 0.3]
                             ,[0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
#%% [markdown]
# 3.5.1 恒等関数とSoftmax関数

#%%
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

#%%
def softmax2(x):
    max_x = np.max(x)
    exp_x = np.exp(x-max_x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

#%%
x = np.array([1,100,1000])
print(softmax(x))
#%%
print(softmax2(x))

#%%
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)


#%%
import pickle
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network2():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network2()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

#%% [markdown]
# 3.6.3 バッチ処理

#%%
batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
