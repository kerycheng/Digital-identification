# 導入函式庫
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils
from matplotlib import pyplot as plt

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

print("train image, size =>", x_train_image.shape)
print("train label =>",y_train_label.shape)
print("test image, size =>",x_test_image.shape)
print("test label =>",y_test_label.shape)

# 建立簡單的線性執行的模型
model = Sequential()
# Input layer
model.add(Dense(units=16, input_dim=784, kernel_initializer='normal', activation='relu'))
# Output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯設定(損失函數, 優化函數, 指標)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# convert label numbers to onehot encoding
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

# convert 2D 28*28 image to 1D 784 array
x_train_2D = x_train_image.reshape(60000, 784).astype('float32') #訓練樣本
x_test_2D = x_test_image.reshape(10000, 784).astype('float32') #測試樣本

# normalize the image numbers to 0~1
x_Train_norm = x_train_2D / 255
x_Test_norm = x_test_2D / 255

# 將訓練過程存入 train_history
train_history = model.fit(x=x_Train_norm, y=y_Train_OneHot, validation_split=0.2, epochs=10, batch_size=50, verbose=2)

# 顯示訓練準確率
scores = model.evaluate(x_Test_norm, y_Test_OneHot)
print()
print("\t[系統訊息] 預測數據準確率 = {:2.1f}%".format(scores[1] * 100.0))

# 顯示前五筆預測數據
X = x_Test_norm[0:5, :]
predictions = model.predict_classes(X)

print(predictions)

# 顯示前五筆預測的圖形
plt.imshow(x_test_image[0])
plt.show()
plt.imshow(x_test_image[1])
plt.show()
plt.imshow(x_test_image[2])
plt.show()
plt.imshow(x_test_image[3])
plt.show()
plt.imshow(x_test_image[4])
plt.show()
