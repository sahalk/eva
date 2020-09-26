# -*- coding: utf-8 -*-
"""Quiz9_Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xxUofd3EJ7_1EsQCqh95CuFIuNfIwGtn
"""

from datetime import datetime
print("Current Date/Time: ", datetime.now())

!git clone "https://github.com/sahalk/eva.git"

from eva.Session9.quiz_loader import Loader
from eva.Session9.QuizDNN import Net

data_mean = (0.4914, 0.4822, 0.4465)
data_std_dev = (0.2470, 0.2435, 0.2616)

cifar = Loader(data_mean, data_std_dev)

cifar.print_sum()

cifar.disp_image(cifar.trainloader)

learning_rate = 0.05
momentum = 0.9
epoch = 10

cifar.train(epoch, learning_rate, learning_rate)