# Machine-Learning-Projects
COEN 240, 2019 Spring, Santa Clara University
--------------------------------------------
## Week 1 Project 1's Contents:
  The Pima Indians diabetes data set (pima-indians-diabetes.xlsx) is a data set used to diagnostically
predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
All patients here are females at least 21 years old of Pima Indian heritage. The dataset consists of M = 8 attributes
and one target variable, Outcome (1 represents diabetes, 0 represents no diabetes). The 8 attributes include
Pregnancies, Glucose, BloodPressure, BMI, insulin level, age, and so on. There are N=768 data samples.
Randomly select n samples from the “diabetes” class and n samples from the “no diabetes” class, and use them
as the training samples. The remaining data samples are the test samples. Build a linear regression model with the
training set, and test your model on the test samples to predict whether or not a test patient has diabetes or not.
Assume the predicted outcome of a test sample is 𝑡̂, if 𝑡̂ ≥ 0.5 (closer to 1), classify it as “diabetes”; if 𝑡̂ < 0.5
(closer to 0), classify it as “no diabetes”. Run 1000 independent experiments, and calculate the prediction accuracy
rate as 𝑡ℎ𝑒 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠 / 𝑡ℎ𝑒 𝑡𝑜𝑡𝑎𝑙 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑡𝑒𝑠𝑡 𝑐𝑎𝑠𝑒𝑠 %. Let n=20, 40, 60, 80, 100, plot the accuracy rate versus n.
--------------------------------------------
## Week 1 Project 2's Contents:
  Iris.xls contains 150 data samples of three Iris categories, labeled by outcome values 0, 1, and 2. Each
data sample has four attributes: sepal length, sepal width, petal length, and petal width.Implement the K-means clustering
algorithm to group the samples into K=3 clusters. Randomly choose three samples as the initial cluster centers. 
Calculate the objective function value J as defined in Problem 3 after the assignment step in each iteration. 
Exit the iterations if the following criterion is met: 𝐽(Iter − 1) − 𝐽(Iter) < ε, where ε = 10−5, 
and Iter is the iteration number. Plot the objective function value J versus the iteration number Iter.
