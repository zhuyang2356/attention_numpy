# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import pandas as pd
import numpy as np


a1=np.linspace(2.0,8.0,num=4)
a1=a1.reshape((4,-1))
print(a1)
Wq=np.linspace(1,16,num=16)
Wq=Wq.reshape((4,4))
print(Wq)
Wk=np.linspace(1,16,num=16)
Wk=Wk.reshape((4,4))

q1=np.matmul(Wq,a1)
k1=np.matmul(Wk,a1)

alpha11=np.matmul(q1.reshape((-1,4)),k1)[0][0]



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
