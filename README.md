# Employee-Attrition
## Objective: develop a prediction model to know if a given employee will be Terminated/Leave or Retained

## Data Cleaning
Initially an Exploratory Data Analysis was performed on the data to understand the structure of the data set. 
Status of the employee was our dependent variable. The employee id column was dropped from the data set.
Further the Age column was distributed into 5 bins of: 18-24, 24-29,30-34 ,35-39 and 40+. 
It was done so that Age does not act as weight in our model. The number of team changes was changed to an integer column by changing 3+ to 4. 
It was done to have a uniform structure.Further all the categorical data was converted using dummy variables for processing through the ML models. 
The data was standardized to make data consistent.

## Exploratory Data Analysis:
An interesting insight was observed of Age with respect to Job Satisfaction. 
A trend was observed that Job Satisfaction falls from 4.0 to 1.4 after age 40.
Intuitively, one would assume that job satisfaction would affect the performance of the employee and thus lead to an effect of Termination.
In addition, it was observed Performance Rating and Job Satisfaction categorized by Number of teams changed. 
EDA reveals that employee who change 0 or 1 team are less satisfied than people who change 2, 3 or more teams. However, the performance of the employee irrespective of the teams changed is almost equally distributed.

## Algorithm Development 
In the project we have compared five different models with 3 different feature selection methodology to compare the results and get the best possible model for prediction. 
 ## Algorithms for comparison 
1) Logistic Regression 2) XG Boost 3) Random Forest 4) SVC 5) KNN
## Metrics Used for Comparison 
1)Accuracy Score, 2) Confusion Matrix 3) Classification report

## Multicollinearity and Excessive columns 
It was identified that there were about 13 columns that projected an issue of multicollinearity using VIF analysis. The values went over 5 which seemed to be a note of caution for our model. Hence, applying the algorithms without completing feature selection would be futile and inefficient. Hence, three different feature engineering methods to eradicate the multicollinearity issue and reduce computational time by reducing the number of insignificant columns. 

## Methodology for feature Selection
1) Backward Elimination 2) Backward Elimination and PCA 3) Backward Elimination, Ensemble Feature Selection and PCA
### 1) Backward Elimination
We applied the trial and error method to attain the best results and accuracy for our model. We tried to create different deployment scenarios based on applying different feature selection techniques for each model. Model 1 follows the results of Backward elimination and algorithms deployed. An OLS Regression was performed to get the P-Values that were significant to our model. Our significance level was decided at 5%.
#### Model Deployment # 1
 We achieved highest accuracy with XG Boost, that was 73%.The precision achieved for XG boost was 73%, which shows that with this model we were able to classify more people in the right category of termination spot as compared to the false positives of predicting them terminated but the employees attrition. Our next focus of analysis was towards the recall rate to identify if we can reduce the number of wrongly attrition employees who are being terminated. This is our main point of focus since our objective was to predict employees that would be terminated. Applying our XG boost we achieved a recall rate of 60%. This is a reasonable result since we were able to rightly classify more True positives as compared to False Negatives. 
### 2) Backward Elimination followed by PCA 
After the model ran on the 48 components, we ran a PCA to select variables which define at least 80% of variability in data. This was done to reduce the model complexity. This allowed us to reduce the dataset from 48 columns to 36 components.
#### Model Development # 2:
Upon performing this step, the data was trained and tested on the 5 models and we could accurately classify 69% of the testing data using XG Boost and SVM. For this model we dropped in our precision to 71% and recall value to 59%. We can analyze that, post applying PCA, we could see a marginal drop in our True positive rate and a slight increase in the False negative numbers. 
### 3) Feature Selection using Backward elimination, Ensemble Extra tree classifier, and PCA
Once we performed Backward Elimination, we accumulated 48 columns, we then tried to reduce the number of columns by using Ensemble Feature selection. We extracted the top 25 features that would have the highest coverage in managing the entropies. Furthermore, we ran a PCA analysis to explain 90% of the data variability on the existing 25 features to reduce the number of variables affecting the prediction of our model. Running a PCA on top 25 significant variables yielded us a result of 14 Principal components to analyze. 
#### Model Development #3:
This method of feature selection use 14 of the selected 25 variables for training our algorithms. The testing of this data set got a maximum accuracy of 68% from SVC algorithm. While the accuracy is not being greatly affected by this model, we are more cautioned by the recall that is being affected, we received a mere 47% for the recall, which shows that due to the variable reduction to 14pcs the machine learning model would have lost significant data to classify terminated employees which is our objective. This model predicted more employees that were placed in the False negative matrix then the other two models deployed previously. 
### 4) Voting Classifier with Grid Search
Once we analyzed the three methods of data processing, we realized our first model involving backward elimination gave us the best results. To further improve the result, we performed a grid search and voting classification using logistic regression, XG Boost and SVM. Upon performing the analysis, we attained an accuracy of 71.1%. Though the accuracy in this model did not improve much from our first model of backward elimination, but the recall rate reduced to 55% from 60%. 
## Conclusion and Tradeoff
Through this attrition data we received numerous columns and multicollinear data, and we tried to understand the most significant features that would affect the model by reducing the number of variables through various feature selection methods. Our feature selection findings showed a glaring insight that was picked by our team, upon reducing the columns we noticed that the columns of JOB_Group created by the dummy variables were highly significant and their influence seemed to be the greatest in all of the models. Hence, if the objective of the user was to only focus on the accuracy, recall and precision scores then the model 1 deployment would be the best choice, since we are obtaining numerous variables and each column is being used to improve the accuracy. However, if the objective of the user is to understand the feature importance and feature influence then we believe that the 3rd model provides sufficient information regarding the most significant columns that account for majority of the accuracy of the model. 

