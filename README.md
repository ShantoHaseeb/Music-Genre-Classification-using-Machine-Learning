Music genre classification is the task of identifying the genre of a music track based on its
features. Genre classification can be useful for a variety of applications, such as music
recommendation systems, playlist generation, and music information retrieval. There are many
different approaches to music genre classification, ranging from simple statistical methods to
more complex machine learning algorithms.

A project on music genre classification could help students and educators understand the
characteristics and conventions of different music genres, improving their appreciation and
understanding of music. By accurately classifying music into different genres, recommendation
systems could better understand a user's preferences and suggest similar music within the same
genre. Classifying music into genres can help with organizing and organizing music libraries,
making it easier to find and listen to specific types of music. By categorizing music into different
genres, it would be easier for listeners to discover new music within a specific genre that they
enjoy. Accurate genre classification could be useful for music industry professionals, such as
music curators and record labels, for tasks such as marketing and promoting specific types of
music.


Link: https://www.kaggle.com/datasets/purumalgi/music-genre-classification?select=train.csv

Reference: The entire credit goes to MachineHack where different hackathons are
hosted for practice and learning.

Column headers/Features, Label, Number of instances/Rows
- The dataset had 17 Features
- It has numerical label
- The dataset has 17,996 data points
- It had 12 unique features.
- Correlation of the features along with the label/class

Correlation between danceability and target: -0.22
Correlation between energy and target: 0.07
Correlation between key and target: -0.02
Correlation between loudness and target: 0.07
Correlation between mode and target: -0.00
Correlation between speechiness and target: -0.26
Correlation between acousticness and target: 0.03
Correlation between instrumentalness and target: -0.04
Correlation between liveness and target: -0.03
Correlation between valence and target: -0.03
Correlation between tempo and target: 0.05

Biasness/Balanced:
- All the classes do not have an equal number of instances.
- Representation using a Histogram, Scatter plot and Correlation Matrix

Data preprocessing:
Before we can apply any classification algorithms to the dataset, we need to preprocess the data.
This may involve cleaning and normalizing the data, as well as extracting relevant features. The
dataset had several Null Values and had to impute the null values using SimpleImputer. I used
LabelEncoder for encoding purposes. Since it had an extra column with no uses I dropped the
“genre” column as I replaced it with the target column later.

Feature Scaling:
To improve the performance of the algorithms, I scaled the features using the min-max scaler.
This helped to ensure that all features were on the same scale, as some features had significantly
larger values than others.

Data Splitting:
I split the dataset into a training set (80%) and a test set (20%) using stratified sampling. This
helped to ensure that the class distribution in the training and test sets was representative of the
overall distribution in the dataset.

Model Training:
I applied total six different algorithms to the dataset (where I have done four mandatory &
two extra for bonuses): support vector machines (SVM), logistic regression, K-Nearest
Neighbor(KNN), Decision Tree Classifier, Gaussian naive Bayes (GNB), and multilayer
perceptron (MLP). For each algorithm, I tuned the hyperparameters using a grid search with
cross-validation to find the optimal values.

I evaluated the performance of the algorithms using the accuracy metric, which measures the
percentage of correct predictions made by the model.

Model selection/Comparison analysis:
Following is provided the bar chart with comparison between different training models. The bar
chart shows that the SVM and logistic regression algorithms performed similarly, with slightly
higher accuracy than GNB. The MLP classifier achieved the highest accuracy among the six
algorithms. 

Accuracy and error rate:
I evaluated the performance of the six different algorithms on the test set and obtained the
following results:
SVC: Accuracy 66.5%, Error Rate 33.5%
Logistic Regression: Accuracy 66.25%, Error Rate 33.75%
KNN: 61% accuracy, Error Rate 39%
Decision Tree: 57.25% accuracy, Error Rate 42.75%
Gaussian Naïve Bayes: Accuracy 27%, Error Rate 73%
NNC: Accuracy 66%, Error Rate 34%

Music genre classification is a challenging task due to the large number of possible genres and
the complexity of the audio signals. However, with the right dataset and classification
algorithms, it is possible to achieve good performance on this task. Further research is needed to
improve the accuracy and robustness of music genre classification algorithms, as well as to
explore new applications for these algorithms.

In future, I would like to deploy my classifier as a web app or mobile app, allowing users to
classify their own music or browse a library of classified songs. But before that I have to
improve the accuracy of our music genre classifier. I will try using a different machine
learning algorithm, fine-tuning the hyperparameters of your existing algorithm, or using a more
sophisticated feature extraction method. I will extend our classifier to handle a wider range of
genres by collecting more data for each genre and training our classifier on this larger dataset. I
may consider incorporating additional features into our classifier, such as lyrics, artist
information, or audio features like tempo and pitch. I can use our trained classifier for other
tasks related to music analysis, such as artist recommendation or playlist generation.
