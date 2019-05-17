# Midterm

#### Project: 1.Climate Simulation Crash

#### Data: 

There are 18 parameters in the data and the output is just 1 number, 1 or 0. Fortunately, we have completely data so we do not need to clean the data-- just use it directly.

The first thing we need to to is to divide the data into training set and test set. There are 540 events in the data so we could regard 380 of 540 as training set and the rest 160 as test set.

Algorithm:

Our goal is to use 18 parameters to calculate one output, 1 or 0 so the best algorithm should be Logistic Algorithm.
$$
h_{\theta}(x) = \frac{1}{1+e^{-\theta^{T}\textbf{X}}}\\\
\theta^{T}\textbf{X} = \theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n
$$
If $h_\theta <0.5$, the output is 0, otherwise, the output is 1.

And the cost function is Cross Entropy function, i.e.
$$
J(\theta) = -\frac{1}{m}[\sum_{i=1}^m(y_ilogh_\theta(x_i)+(1-y_i)log(1-h_\theta(x_i)))]
$$
Our goal is to find a $\theta$ which can minimize the $J(\theta)$.

To find the $\theta$, we use Gradient Descent Algorithm which mean use the following function to calculate it:
$$
\theta_j=\theta_j - \alpha(\frac{\partial}{\partial\theta_j}J(\theta))=\theta_j-\alpha(\frac{1}{m})\sum_{i=1}^{m}(h_\theta(x^i)-y^i)x^{i}_j
$$
Following is my code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def logit(x,w):
    return 1 / ( 1 + math.exp( -np.dot( w[0,0:18] , x[2:20] ) - w[0,18] ) )

def stLogit(x):
    return x[20]

def costFunc(x,w):
    sum = 0
    for i in range(0,len(x[:,1])):
        sum += - ( stLogit(x[i]) * math.log( logit(x[i],w) ) + ( 1 - stLogit(x[i]) ) * math.log(1-logit(x[i],w) ) ) 
    return(sum/len(x[:,1]))

def GDA(x,w_temp,i,a):
    sum = 0
    if i == 18:
        for j in range(0,len(x[:,1])):
            sum += ( logit(x[j],w_temp) - stLogit(x[j]) ) * 1
    else: 
        for j in range(0,len(x[:,1])):
            sum += ( logit(x[j],w_temp) - stLogit(x[j]) ) * x[j,i+2] 
    w_temp[0,i] = w_temp[0,i] - a  * sum / len(x[:,1])
    return w_temp[0,i],sum

def walk(x,a):
    w = np.zeros((1,19),dtype=float)
    w_temp = np.zeros((1,19),dtype=float)
    cost = 10
    w_best = np.zeros((1,19),dtype=float)
    for j in range(0,10):
        flag = True
        for i in range(0,19):
            while True:
                w = w_temp
                w_temp[0,i] ,sum= GDA(x,w_temp,i,a)
                temp = costFunc(x,w_temp)
                if (temp < cost) :                
                    cost = temp
                    w_best = w_temp
                if abs( sum/len(x[:,1]) ) <= 0.001 :
                    break
    return w_best
```



logit function is used to calculate the $h_\theta$(x) under a certain $\theta$ and stLogit is used to calculate the $y_i$.

costFunction is used to calculate the Cost Function $J(\theta)$.

GDA is used to give the next $\theta$ according to the GDA function.

walk function can give the best $\theta$.



There is a parameter which can affect the performance of our algorithm--$\alpha$. In my code, it is "a". To find the best $\alpha$, I run the following code:

```python
min_a= 0
min_w=np.ones((1,19))
w = np.ones((1,19))
rate = 1
for a in range(100,200,10):
    i = a/1000.0
    w = walk(x,i)
    if (costFunc(x,w) < rate):
        min_w = w
        min_a = i
        rate = costFunc(x,w)
min_a
```

The output is 0.19. Then I use $\alpha=0.19$ to evaluate our code.

```python
def judge(x,w,threshold):
    if logit(x,w) > threshold :
        if stLogit(x) == 1:
            return True
        else:
            return False
    if logit(x,w) < threshold:
        if stLogit(x) == 0:
            return True
        else:
            return False 
        
def Judge(x,w,threshold):   
    counter = 0
    for i in range(0,len(x[:,1])):
        if judge(x[i],w,threshold) == True:
            counter += 1
    return (counter/len(x[:,1]))
```

This function can give us the correct rate of our code.

$\theta$ (the last one is $\theta_0$): 

```python
array([[-1.42739419e+00, -6.84830971e-01,  1.32003949e+00,
         1.52707817e+00,  1.00251737e+00,  9.95684799e-01,
         3.98074402e-01,  9.35229658e-03,  1.34414556e-02,
         9.79908099e-03,  1.46604196e-02,  6.45100989e-05,
        -8.09271828e-02,  1.66623642e+00,  5.15601201e-03,
         3.33804038e-01,  1.22408902e-02, -1.30059619e-04,
         1.82230148e-02]])
```



On the training set, the correct rate is 93.2%. And on the test set, the correct rate is 90.6%. But there is a problem: If the event is successful, the algorithm can give 1 100% while if the event is failed, it just can give 0 about 34%.

So I try some way to solve it. I just use the events which are failed to train my code and then get a new $\theta$(the last one is $\theta_0$):

```python
array([[-1.02085353e+01, -2.44778141e+00, -1.86501680e+00,
        -5.00586445e-05, -1.21181523e-05, -6.12490057e-05,
        -1.91697287e-04, -1.14593280e-04, -1.40668743e-04,
        -2.30695280e-01, -1.33045709e-04, -6.58327894e-05,
        -2.04132708e-04, -2.72537175e-04, -8.02890932e-06,
        -1.58142651e-05, -2.36502610e-05, -2.15164849e-04,
        -3.06827216e+00]])
```

Under this $\theta$, if the event is failed, it can give 0 on 97.9%. But under this $\theta$, if the events is successfully, it can give 1 on about 10%.

So the best way to solve this problem is use 2 different $\theta$, give us the probability.

Summary:

With the first $\theta$, if the event is successful, we can give output 1 100%. With the second $\theta$, if the event is failed, we can give output0 98%. The best way to solve this problem is to combine two $\theta$ to calculate the probability.

