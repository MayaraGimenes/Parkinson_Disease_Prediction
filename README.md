<!-- PROJECT LOGO -->
<br />

  <h1 align="center">Parkinson's Disease Prediction Based on <br />Modern Machine Learning Methods</h1>

  <p align="center">
    Mayara Gimenes De Souza <br />
    Diego Andres Roa Perdomo
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Key-words">Key Words</a></li>
      </ul>
    </li>
    <li><a href="#Introduction">Introduction</a></li>
    <li><a href="#Related-Work">Related Work</a></li>
    <li><a href="#Methods">Methods</a>
    <ul>
        <li><a href="#Feature-Calculation">Feature Calculation</a></li>
        <li><a href="#The-Dataset">The Dataset</a></li>
        <li><a href="#Support-Vector-Machine">Support Vector Machine</a></li>
        <li><a href="#Random-Forest">Random Forest</a></li>
        <li><a href="#Neural-Networks">Neural Networks</a></li>
        <li><a href="#XGBoost">XGBoost</a></li> 
        <li><a href="#Naive-Bayes">Naive Bayes</a></li> 
        <li><a href="#TPOT-AutoML">TPOT AutoML</a></li> 
      </ul>
      </li>
    <li><a href="#Implementation">Implementation</a>
    <li><a href="#Results">Results</a>
    <ul>
        <li><a href="#SVM">SVM</a></li>
        <li><a href="#NN">NN</a></li>
        <li><a href="#RF">RF</a></li>
        <li><a href="#XGB">XGB</a></li> 
        <li><a href="#NB">Naive Bayes</a></li> 
        <li><a href="#TPOT">TPOT</a></li> 
      </ul>
    </li>
    <li><a href="#Futuree-Work">Future Work</a></li>
    <li><a href="#Conclusion">Conclusion</a></li>
    <li><a href="#References">References</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this report, we present an approach of Parkinson's disease(PD) prediction
using modern machine learning techniques such as Support Vector Machine (SVM),
Neural Networks, XGBoost, Random Forest, Naive Bayes and automatic ML generation 
with TPOT. We used a dataset available online with voice variations of health 
individuals and patients with PD. Through correlated measures, we selected the 
10 most correlated features, then we evaluated the model using the k-fold cross 
validation method leading to the best accuracy of 89.65%
using kernel support vector machine and XGBoost.

### Key Words

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Python](https://www.python.org)
* [scikit-learn](https://scikit-learn.org/stable/)


<!-- GETTING STARTED -->
## Introduction

Parkinson's disease (PD) is a neurodegenerative disorder of the central nervous system
that affects movements and develops slowly over the years and the earlier you discover
it the best results the patients may have doing the recommended treatment. PD affects
almost a million people in North America. Currently there is no cure, however there are
several types of medication and therapy that may alleviate the symptoms. Research has
proven that approximately 90% of the people with Parkinson's has voice impairment
[6], [7]. This distortion on the voice maybe be an early stage of the illness [8] and its
measurement is a non invasive way to predict the probability of PD. To determine
if the patient presents voice impairment it is necessary to perform various sets of
tests including sustained phonations. All the speech sounds are freely available on the
internet, and they were recorded using a microphone and analyzed using measurement
methods to detect these signals [9]. All the results and parameters measured from the
vocal impairments are listed in the Table 1.

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/table1.JPG)

### Related Work

This project is based on the paper "Exploiting Nonlinear Recurrence and Fractal Scal-
ing Properties for Voice Disorder Detection', Little MA, McSharry PE, Roberts SJ,
Costello DAE, Moroz IM", where they present a study using nonstandard measures
(Dysphonia measurements) to discriminate healthy people from patients with Parkin-
son's disease. The authors used the same dataset, a collection of phonations from 
31 people, and 23 with PD. In their work they used the most uncorrelated measures to
train the system, achieving a performance of 91.4% using kernel vector machine. The
authors use bootstrap re-sampling as a cross validation method, whereas this work
uses the 10-fold cross validation, therefore the results are not directly comparable.
The k-fold cross validation was chosen due to its popularity and reliability.
This report focuses on the work of using the same dataset, but applying three more
classification methods (XGBoost, Neural Network, Random Forest) in addition to the
Support Vector Machine (SVM), which was implemented on the mentioned paper.

## Methods

### Feature Calculation
In the preprocessing stage, the data is filtered in order to decrease the dimensionality
of the problem and therefore decrease the training and prediction complexity (from 24
features, we choose to use only 10). In the case of SVM and XGBoost, we choose to
train all possible 210 combinations and obtained the best feature combination, since
these methods don't provide hyper-parameters that can be tuned during training. For
the other scenarios (Random Forest and Neural Networks), we choose to use all 10
features and change the hyper parameters from each model. In order to choose 10 out
of the 24 features, we notice that many features are correlated to each other, since
they have similar aspects. On table 1 we can verify that Jitter(%) is very similar from
Jitter(ABS), they derive from pitch period sequences, and because of their similarity
(correlation) only one of them will really contribute to the classification, consequently
the other one should be removed.

We divide the calculation of feature by selecting the 10 most correlated features (absolute 
value of correlation between feature and output label). We sort the correlation
based on its absolute value because zero correlation means almost no information,
whereas -1.0 or +1.0 mean the same amount of information, however, in different directions.
Analysing carefully through all features and calculating all the correlations, we
selected the 10 most correlated in absolute value. For instance, if after calculating the
correlation we get [0.2, 0.3, 0.15, -0.5, -0.25] the most three correlated numbers will
be [-0.5, 0.3 and -0.25] and these will be selected to the training and test set. In the
next sections we will describe the classifications tested on this work. All testing and
plots of the data and correlation between each feature and the label can be seem on
figure 1. All 10 features selected are shown on Table 1.

Evaluating the model can be really trick, since we usually split the data set into
training and testing and we evaluate the performance based on a error metric to
determine the accuracy. However this method is not reliable enough with the amount
of data provided. If we apply the k-fold cross validation method we may increase the
reliability of the accuracy, avoiding overfitting the data. It basically divides each data
into folds and ensures that each fold is used as a test. For the K-fold, the data is split
into K folds, in our case we decided to use K=10, so our data set is split into 10 folds.
In the first iteration, the first fold is used to to test the model and the rest to train
the model; in the second iteration the second fold is used to test the model and the
rest is used to train the model, and this process repeat for each fold until we have all
10 tested. The accuracy of each fold is averaged and that is the final performance of
the model.

To implement all the classifications on this work, we used the scikit-learn features
for Python [1]. Scikit-learn is an open source machine learning library that supports
supervised and unsupervised learning and provides tools for model fitting, model se-
lection and evaluation. 

### The Dataset
The dataset[5] used is composed of a range of biomedical voice measurements from 31
people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice
measure, and each row corresponds one of 195 voice recording from these individuals
("name" column). The main goal of the data is to discriminate healthy people from
those with PD, according to "status" column which is set to 0 for healthy and 1 for
PD.

The data is in ASCII CSV format. The rows of the CSV file contain an instance
corresponding to one voice recording. There are around six recordings per patient, the
name of the patient is identified in the first column.

### Support Vector Machine
For each combination selected we will apply SVM classification. SVM is a supervised
machine learning model that uses classification algorithms for a group of classification
models. SVM offers high accuracy compared to other classifiers and there are some
kernel tricks to handle non linear input spaces. The main advantages of this method
are its efficiency in high dimensional spaces, memory eficiency, and simplicity.
SVM constructs a hyper-plane in multidimensional space to separate different
classes. It optimizes this hyper-plane to minimize the error and best divide the data
sets.

The main objective here was to separate the data set the best possible way by
selecting a hyper-plane with the maximum possible margin between the vectors and
the data points. The algorithm is implemented using a kernel that will shape the
given data on the correct form. In this work, the best option would be the radial basis
function kernel (also known as Gaussian kernel). It can map an input space in "infinite
dimensional space".

### Random Forest
Random forest [2] is another supervised learning algorithm used in classification problems. 
It creates decision trees on random data samples, receives a prediction from each
tree and select the best solution. It also adds additionally randomness to the selected
model and it searches for the best feature among our subset, this way it generally
results in better models. The difference between random forest and decision trees is
that the process of finding the root node and splitting the features will be a random
process on random forests. First we create the random forest and then we make the a
prediction of the classifiers. The algorithm is shown below:

Select N features from a total of Z features
* Of all the selected N features it will calculate a node d using the best split point
* Split nodes into more nodes using the best split point
* Repeat the items above until the best number D of nodes has been reached
* Build the forest repeating the steps for x number of times to create x number of
trees

There were some important parameter to check while implementing the algorithm. One
of those is the number of trees we want to implement, so the average of predictions
was measured multiple times guaranteeing that we would have the optimal number of
trees.

The biggest advantages of random forest is its versatility, since it can be used both
for regression and classification, the hyper-parameter that are pretty straightforward
to use. Nevertheless, the biggest problem with this classifier is that a large number of
trees can make the algorithm really slow.

### Neural Networks
Neural networks are graph models inspired in biological neural networks present on
the human brain. It is built by an ensemble of nodes called perceptrons, which sum
the output of the nodes connected to them and get "activated" (or not) depending
on the value on their input. In order to get activated, each perceptron depends on
the weights of the connections with its inputs, as well as on the activation function
chosen for a given model. The network is arranged in a way that layers of perceptrons
are constructed and trained in order to decrease the error between estimated output
and test output.

As the error decreases, it should also be able to generalize well for
unknown inputs (test data). In this work we implemented a multi layer perceptron
using backpropagation, a procedure that adjusts the weights repeatedly so we can
minimize the diffence between the output and the desired output. We also test
repeatedly several numbers of hidden layers so it can achieve the best accuracy.
We set the parameters of our NN to get the best value. To optimize our network we
chose as our solver the Adam Algorithm [3], a very popular deep learning optimization method. 
The activation function is responsible for transforming the sum of the inputs
from the nodes into the activation of the node. In this work, we explore the rectifi
linear activation function (relu - a linear function that will output the input if positive,
otherwise, will output zero), hyperbolic tangent, and logistic function.

### XGBoost
XGBoost [4] is an optimized gradient boosting library designed to be efficient, 
exible and portable. It implements machine learning algorithms under the gradient boosting
framework. A gradient boosting framework is a technique for regression and classification 
problems, that produces a prediction model in the form of decision trees. New
models are created predicting residual errors of the prior models, eventually this errors
are added to build the final prediction and it uses gradient descendent to minimize the
when adding new models (also known as weak model algorithm). Its algorithm was
created to reach for efficiency of compute time and memory resources.

### Naive Bayes
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ 
theorem with the “naive” assumption of conditional independence between every pair of 
features given the value of the class variable. Abstractly, naive Bayes is a conditional 
probability model, features are represented by a vector X = (x1, x2, ..., xn). The model 
assigns probabilities p(Ck | x1, x2, ..., xn) for every k output. In this case we calculate
only one output, corresponding to the diagnosis of the patient. Using Bayes' theorem, the 
conditional probability can be decomposed as:

p(Ck|x) = p(Ck) * p(x|Ck) / p(x)

or:

Posterior = Prior * Likelihood / Evidence

In spite of their apparently over-simplified assumptions, naive Bayes classifiers 
have worked quite well in many real-world situations, famously document classification 
and spam filtering. They require a small amount of training data to estimate the necessary 
parameters. In this case we applied Gaussian Naive Bayes, assuming that that the continuous 
values associated with each class are distributed according to a normal (or Gaussian) 
istribution. Despite the fact that the far-reaching independence assumptions are often 
inaccurate, the naive Bayes classifier has several properties that make it surprisingly 
useful in practice. In particular, the decoupling of the class conditional feature 
distributions means that each distribution can be independently estimated as a one-dimensional 
distribution. This helps alleviate problems stemming from the curse of dimensionality, 
such as the need for data sets that scale exponentially with the number of features. 

### TPOT AutoM

TPOT is an open-source library for performing AutoML in Python. It makes use of the popular 
Scikit-Learn machine learning library for data transforms and machine learning algorithms 
and uses a Genetic Programming stochastic global search procedure to efficiently discover 
a top-performing model pipeline for a given dataset. TPOT uses techniques for automatically 
discovering well-performing models for predictive modeling tasks with very little user
involvement. In short, TPOT optimizes machine learning pipelines using a version of genetic 
programming (GP), a well-known evolutionary computation technique for automatically 
constructing computer programs. The GP algorithm generates 100 random tree-based pipelines 
and evaluates their balanced cross-validation accuracy on the data set. For every generation 
of the GP algorithm, the algorithm selects the top 20 pipelines in the population according 
to the NSGA-II selection scheme.  Each of the top 20 selected pipelines produce five copies 
(i.e., offspring) into the next generation’s population, 5% of those offspring cross over 
with another offspring using one-point crossover, then 90% of the remaining unaffected 
offspring are randomly changed by a point, insert, or shrink mutation (1/3 chance of each).

## Implementation
We have implemented our project using Google Colab, this allows for others to run the 
program with minimal setup Colab notebooks allow you to combine executable code and rich 
text in a single document. Colab allows anybody to write and execute arbitrary python code 
through the browser, and is especially well suited to machine learning, data analysis and 
education. This [Notebook](https://colab.research.google.com/drive/1fxLPRrO-8_fBCmX4jUaepxO2V_S1qkmo?usp=sharing) [7]

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/ColabN.png)

<ol>
<li>Check that the Colab instance is connected and running, a green checkmark indicates that everything is in order</li>
<li>You can "Run all" to execute the notebook</li>
<li>The generated plots are stored in the directory</li>
<li>It is also possible to run independent cells of the notebook</li>
</ol>

## Results 

In this section we present results for the test accuracy of each model by using 10-fold
crossvalidation. By accuracy we mean the percentage of labels that were correctly
classified. 

### SVM
In the Figure 2, we present the accuracy of the SVM model with Radial Basis Function 
as a kernel. The "Selected Features Index" axis represents all the possible 210
combinations of selected features. The maximum accuracy obtained is 85.89% by using 
two features, PPE MDVPShimmer and PPE ShimmerAPQ5. It can be seen thatdepending on 
the selection, the accuracy drastically changes, therefore, it is important to make 
a proper selection of features.

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/SVM.png)

Figure 2.: SVM accuracy vs Selected features index

### NN
In the Figure 3 we have the results of using the Logistic Neural Network checking for
the accuracy using different values of hidden nodes in 2 layers. The highest accuracy
is 82.36% with 15 hidden nodes in Layer 1 and 20 hidden nodes in Layer 2.
For ReLu Neural Network (Figure 4) the maximum accuracy is 83.39% with 35
hidden nodes in Layer 1 and 40 hidden nodes in layer 2. This result is followed by an
accuracy of 81.842% with 20 hidden nodes in layer 1 and 5 hidden nodes in layer 2. It
is interesting to see that the result changes considerably depending on the amount of
hidden nodes on each layer.
For Tahn activation, best accuracy of 81.97% for 10 hidden nodes in layer 1 and 20
hidden nodes in layer 2.
The best performing neural network uses the logistic function with 15 hidden nodes
in Layer 1 and 20 hidden nodes in Layer 2, with an accuracy of 82.36%.

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/LogiNN.png)

Figure 3.: Logistic Neural Network.
(a) Accuracy with different numbers of hidden layers

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/ReluNN.png)

Figure 4.: ReLu Neural Network Accuracy.
(a) Accuracy with different number of hidden nodes


![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/TanhNN.png)

Figure 5.: Tahn Neural Network Accuracy
(a) Accuracy with different number of hidden nodes

### RF
We showe the results for the random forest case in the Figure 6. The hyper parameter
selected to be tuned is the number of estimators used. In this case, the best value of
accuracy obtained is 85.47% with 400 estimators.

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/RF.png)

Figure 6.: Random Forest accuracy vs Number of Estimators

### XGB
Figure 7 shows that for XGBoost method we achieved an accuracy of 89.657% with
the following features Spread1, Spread2 and MDVPFo.

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/XGBoost.png)

Figure 7.: XGBoost accuracy vs Selected Features Index

### NB
The program generates Naive Bayes models based on the combinatory of the selected features. 
Starting with one feature, the program increases the number of features and test all possible 
combinatons. Figure 8 shows the performance of each model, the best model with an accuracy of 
84.39% uses only one feature RPDE, the feature with the highest correlation. 

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/NaiveBayes.png)

Figure 8.: Naive Bayes accuracy vs Selected Features Index

### TPOT
We define the classifier to have a hundred generation with a population of a hundred each
to generate the different pipeline models. The best pipeline after the execution of the program 
is Gradient Boosting Classifier, with an accuracy of 85.25%.

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/TPOT.png)

Figure 9.: TPOT Accuracy vs Number of Generations

This repository includes the pipeline generated after a hundred generations, the program
"tpot_pipeline_G100.py" is also one of the outputs of TPOT.

## Future Work
Although this project was meant to be a practical experience using ML tools, this analysis 
can be used to develop hybrid models based on those that have performed the best or to 
further tune them and improve their performance. It is also important to point out that the 
database is small, limiting the reach of the models. This kind of ML applications can be used 
to develop fast and innexpensive ways for early diagnosis, allowing the implementation in 
devices like smartphones. Once the model is trained, the inference phase can be carried out 
in a less powerful device.

## Conclusion
In the Table 2 we can see the best performance obtained by each machine learn-
ing method. There is a considerably difference between the methods, as well as the
complexity of finding good values. It is interesting to see that two best methods (XG-
Boost and SVM) have their peak performance with by using different features. We can
conclude that among all modern machine learning methods presented on this paper,
XGBoost had the best performance overall. That method has accuracy of 89.65%,
meaning that there is a probability of almost 10% for the system to making a mistake
on the classification if a person has Parkinson's disease. While this might seen high,
it is very close to the one presented by the paper that inspired this work, even though
we use a different cross validation technique (k-fold).

Table 2.: Comparison of best accuracy obtained by each model

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/Results2.png)

Figure 10.: Comparison of best accuracy obtained by each model

![](https://github.com/MayaraGimenes/CISC849_FinalProject/blob/main/Pictures/Results.png)


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/MayaraGimenes/CISC849_FinalProject/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 


<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE` for more information. -->



<!-- CONTACT -->
<!-- ## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name) -->



<!-- ACKNOWLEDGEMENTS
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com) -->





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png



