



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
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
        <li><a href="#Support-Vector-Machine">SVM - Support Vector Machine</a></li>
        <li><a href="#Random-Forest">Random Forest</a></li>
        <li><a href="#Neural-Networks">Neural Networks</a></li>
        <li><a href="#XGBoost">XGBoost</a></li> 
      </ul>
      </li>
    <li><a href="#Results">Results</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

In this report, we present an approach of Parkinson's disease(PD) prediction
using modern machine learning techniques such as Support Vector Machine (SVM),
Neural Networks, XGBoost and Random Forest. We used a dataset available online
with voice variations of health individuals and patients with PD. Through correlated
measures, we selected the 10 most correlated features, then we evaluated the model
using the k-fold cross validation method leading to the best accuracy of 89.65%
using kernel support vector machine and XGBoost.

### Key Words

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Python](https://www.python.org)
* [scikit-learn](https://scikit-learn.org/stable/)
* [XGBoost] ()
* [Support Vector Machine]() 


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
son's disease. The authors used the same dataset, a collection of phonations from 31 people, and 23 with PD. In their work they used the most uncorrelated measures to
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
they have similar aspects. On table I we can verify that Jitter(%) is very similar from
Jitter(ABS), they derive from pitch period sequences, and because of their similarity
(correlation) only one of them will really contribute to the classification, consequently
the other one should be removed.

We divide the calculation of feature by selecting the 10 most correlated features (ab-
solute value of correlation between feature and output label). We sort the correlation
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
reliability of the accuracy, avoiding over-fitting the data. It basically divides each data
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
lection and evaluation. To evaluate the data I decided to use four classification models
that are basically the state-of-art in machine learning techniques: support vector ma-
chine, random forest, neural network, and XGBoost.

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

### SVM - Support Vector Machine
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
function kernel (also known as Gaussian kernel). It can map an input space in innite
dimensional space.

### Random Forest
Random forest [2] is another supervised learning algorithm used in classification prob-
lems. It creates decision trees on random data samples, receives a prediction from each
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

There were some important parameter to check while implementing the algorithm.One
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
chose as our solver the Adam Algorithm [3], a very popular deep learning optimization method. The activation function is responsible for transforming the sum of the inputs
from the nodes into the activation of the node. In this work, we explore the rectifi
linear activation function (relu - a linear function that will output the input if positive,
otherwise, will output zero), hyperbolic tangent, and logistic function.

### XGBoost
XGBoost [4] is an optimized gradient boosting library designed to be efficient, 
exible nd portable. It implements machine learning algorithms under the gradient boosting
framework. A gradient boosting framework is a technique for regression and classification problems, that produces a prediction model in the form of decision trees. New
models are created predicting residual errors of the prior models, eventually this errors
are added to build the final prediction and it uses gradient descendent to minimize the
when adding new models (also known as weak model algorithm). Its algorithm was
created to reach for efficiency of compute time and memory resources.

## Results 

In this section we present results for the test accuracy of each model by using 10-fold
crossvalidation. By accuracy we mean the percentage of labels that were correctly
classified.



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
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
* [Font Awesome](https://fontawesome.com)





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
