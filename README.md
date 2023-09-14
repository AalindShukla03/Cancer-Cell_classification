# Cancer-Cell_classification

<h2>Objective :</h2>
<p>To classify cancer cells based on their features, and identifying them if they are ‘malignant’ or ‘benign’. We will be using scikit-learn for a machine learning problem. Scikit-learn is an open-source machine learning, data mining and data analysis library for Python programming language.</p>

<h2>About the Project :</h2>
<p>In this Python Project, Scikit-learn comes with a few small standard datasets that do not require downloading any file from any external website. The dataset that we will be using for our machine learning problem is the Breast cancer wisconsin (diagnostic) dataset. The dataset includes several data about the breast cancer tumors along with the classifications labels, viz., malignant or benign.</p>

<h2>Dataset :</h2>
<p>The data set has 569 instances or data of 569 tumors and includes data on 30 attributes or features like the radius, texture, perimeter, area, etc. of a tumor. We will be using these features to train our model.</p>
<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>scikit-learn</li>
  
       pip install scikit-learn
</ul>
<ul>
 <li>jupyter</li>
  
       pip install jupyter
</ul>

<h2>Step by step implementation of classification using Scikit-learn :</h2>
 <ul>
  <li> 
    
# importing the Python module
    import sklearn

# importing the dataset
    from sklearn.datasets import load_breast_cancer
# loading the dataset
    data = load_breast_cancer()
# Organize our data
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']
# looking at the data
    print(label_names)
<strong>Output:</strong>
 ['malignant' 'benign']
# To Print the labels:-
    print(labels)
<pre>[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
 1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
 1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
 1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
 0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 0 0 0 0 0 1]</pre>
 <p>From here, we see that each label is linked to binary values of 0 and 1, where 0 represents malignant tumors and 1 represents benign tumors.&nbsp;<br>&nbsp;</p>

# To Print the feature names:-
    print(feature_names)
<strong>Output:</strong>
<pre>['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']</pre>
 <p>Here, we see all the 30 features or attributes that each dataset of the tumor has. We will be using the numerical values of these features in training our model and make the correct prediction, whether or not a tumor is malignant or benign, based on these features.&nbsp;<br>&nbsp;</p>
 
 # To Print the labels:-
    print(feature_names)
<strong>Output</strong>

<pre>[[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
 [2.057e+01 1.777e+01 1.329e+02 ... 1.860e-01 2.750e-01 8.902e-02]
 [1.969e+01 2.125e+01 1.300e+02 ... 2.430e-01 3.613e-01 8.758e-02]
 ...
 [1.660e+01 2.808e+01 1.083e+02 ... 1.418e-01 2.218e-01 7.820e-02]
 [2.060e+01 2.933e+01 1.401e+02 ... 2.650e-01 4.087e-01 1.240e-01]
 [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02]]</pre>
 <p>This is a huge dataset containing the numerical values of the 30 attributes of all the 569 instances of tumor data.<br>So, from the above data, we can conclude that the first instance of tumor is malignant and it has a mean radius of value 1.79900000e+01.&nbsp;<br>&nbsp;&nbsp;<br><strong>Step #4:</strong> Organizing the data into Sets.<br>For testing the accuracy of our classifier, we must test the model on unseen data. So, before building the model, we will split our data into two sets, viz., training set and test set. We will be using the training set to train and evaluate the model and then use the trained model to make predictions on the unseen test set.&nbsp;<br>The sklearn module has a built-in function called the <strong>train_test_split()</strong>, which automatically divides the data into these sets. We will be using this function to split the data.&nbsp;<br>&nbsp;</p>

 # TRAIN TEST SPLIT:-
 <p>For testing the accuracy of our classifier, we must test the model on unseen data. So, before building the model, we will split our data into two sets, viz., training set and test set. We will be using the training set to train and evaluate the model and then use the trained model to make predictions on the unseen test set. 
The sklearn module has a built-in function called the train_test_split(), which automatically divides the data into these sets. We will be using this function to split the data. </p>

# importing the function
    from sklearn.model_selection import train_test_split

# splitting the data
    train, test, train_labels, test_labels = train_test_split(features, labels,
									test_size = 0.33, random_state = 42)

<p>The train_test_split() function randomly splits the data using the parameter test_size. What we have done here is that we have split 33% of the original data into test data (test). The remaining data (train) is the training data. Also, we have respective labels for both the train variables and test variables, i.e. train_labels and test_labels.</p>

# Building the Model.
<p>There are many machine learning models to choose from. All of them have their own advantages and disadvantages. For this model, we will be using the Naive Bayes algorithm that usually performs well in binary classification tasks. Firstly, import the GaussianNB module and initialize it using the GaussianNB() function. Then train the model by fitting it to the data in the dataset using the fit() method.</p>

# importing the module of the machine learning model
from sklearn.naive_bayes import GaussianNB

# initializing the classifier
    gnb = GaussianNB()

# training the classifier
    model = gnb.fit(train, train_labels)

<p>After the training is complete, we can use the trained model to make predictions on our test set that we have prepared before. To do that, we will use the built-in predict() function which returns an array of prediction values for data instance in the test set. We will then print our predictions using the print() function.</p> 

# making the predictions
    predictions = gnb.predict(test)

# printing the predictions
    print(predictions)
<strong>Output :</strong>
<pre>[1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0
 1 1 0 0 0 1 1 1 0 0 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 1 1 0 0 1 0 1 1 0 1 0 0
 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 0 1 1 0 1 1 1 1 1 1 0 0
 0 1 1]</pre>
 <p>From the output above, we see that the predict() function returned an array of 0s and 1s. These values represent the predicted values of the test set for the tumor class (malignant or benign). </p>

 # Evaluating the trained model’s accuracy.
<p>As we have predicted values now, we can evaluate our model’s accuracy by comparing it with the actual labels of the test set, i.e., comparing predictions with test_labels. For this purpose, we will be using the built-in accuracy_score() function in the sklearn module. </p>

# importing the accuracy measuring function
    from sklearn.metrics import accuracy_score

# evaluating the accuracy
    print(accuracy_score(test_labels, predictions))
<strong>Output :</strong>
0.9414893617021277
<p>So, we find out that this machine learning classifier based on the Naive Bayes algorithm is 94.15% accurate in predicting whether a tumor is malignant or benign.<br>&nbsp;</p>



    

</li>
  
   
