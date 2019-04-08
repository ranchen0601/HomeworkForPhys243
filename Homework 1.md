<font size=10>Homework 1:</font>

*********

<font size =6>Problem 1:</font>

part 1:

```python
def Fib_loog(n=0):
    sum = 0
    for i in range(0,n):
        sum+=i
    return sum
```

part 2:

​	Compared with the recursive one, function which use loop will run faster because for recursive, it must calculate it floor by floor and it will cost a lot of time to create each floor then go back to the first floor. For example, it is just like go upstairs to the highest floor and then go down to the ground to get the result but for function which use loop, it just calculate it step by step and then give the result directly. Also, the recursive one will cost much more memory compared with the loop one.

part 3:

```python
import time
def Fib_rec(n=0):
    if n==0:
        return 1
    if n==1:
        return 1
    else:
        return Fib_rec(n-1) + Fib_rec(n-2)
def timer(k,n=0,f=Fib_rec):
    sum = 0
    for i in range(0,k):
        x = time.time()
        f(n)
        x = time.time() - x
        sum += x
    average = sum/k
    print(average)()
```

part 4:

![](C:\Users\ranch\Desktop\python\PHYS243\AverageTime.jpg )

******

<font size =6>Problem 2:</font>

![](C:\Users\ranch\Desktop\python\PHYS243\0001.jpg)

![](C:\Users\ranch\Desktop\python\PHYS243\0002.jpg)

*****

<font size=6>Problem 3:</font>

part 1:

​	The validation samples  are used to tune the parameters of the algorithm and the test samples are used to test the algorithm. Which means, the validation samples are used to create the algorithm while the test samples are used to value the algorithm whether useful or not.

part 2:

​	Supervised learning algorithm: What we have is a lot of labeled data and what we want to do is to make some predictions. For example, we have some classified images and we also have some unclassified images so we can use supervised learning algorithm to classify those images.

​	Unsupervised learning algorithm: We don't have any data with label so the algorithm need to learn how to label those data by it self. For example, we have some news which we don't know their classification, so we need the unsupervised learning algorithm to help us to put them into different class.

​	The most important difference is whether the input data with label or not. If input is labeled data, it should be supervised learning algorithm, or not, unsupervised learning algorithm.