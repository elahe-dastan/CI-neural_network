The aim of this homework is to know how perceptrons and neural networks work better, I'm going to report what I did <br/>
step by step <br/>

#### 1. Scattering data
First of all I want to scatter the data and show each group with a different color. <br/>
This is what I got<br/>
![](Figures/Figure_1.png)<br/>
red color is used to show label 1 and blue is used to show 0<br/>

#### 2. Splitting data
Then I splitted the data as 80% training and 20% test data

#### 3. Perceptron
In this part I want to implement a single perceptron, the cost function, the derivatives and the activation function I<br/>
used are shown below <br/>
![](Figures/derivation1.jpg) 

#### 4. Train perceptron
Now the perceptron is trained by training data and then tested, the important thing to do in this part is to give <br/>
appropriate number of epochs and learning rate to perceptron and to do so I had to examine different amounts and choose<br/>
one at the end.

#### 5. Scattering test data
All the data we have is shown in part [1](#1-scattering-data), the image below is the result of the perceptron for test data <br/>
(Keep in mind that this picture is just for one random run and it changes every time I run the code)<br/>
![](Figures/Figure_1-1.png)

#### 6. More layers
Up to now I have implemented a single perceptron, now it's time to have multiple of them together. What I want to <br/>
implement and the derivations I need are shown below <br/>
![](Figures/derivations2.jpg)

#### 7. Train neural network 
Just like part [4](#4-train-perceptron) I examined different learning rates and number of epochs and chose a relatively <br/>
good one for both and then I tested the trained network like part [5](#5-scattering-test-data)<br/>
![](Figures/Figure_1-2.png)

#### Points and conclusion
First of all consider that this sample is small and I cannot make a comprehensive conclusion but I here using <br/>
two layers of perceptrons had a better result than just one perceptron and WHY?!:thinking:<br/>
One perceptron can just classify data into two parts in other words it draws one line so if the data we have can be <br/>
separated nicely using a straight line then just one perceptron is enough but the data we have needs at least two straight <br/>
lines to do a good separation so we need two perceptrons in the first layer to draw those two lines and a third one to <br/>
merge those two.

If what I concluded was right so for a figure like below we need at least four straight lines <br/>
and as a result four perceptrons in the first layer<br/>
![](Figures/question.jpg)