# SQL-Injection--Detection-using-Machine-Learning

Introduction : 
1. SQL Injection also called as SQLI queries. It is a technique where the 
attacker creates or alters SQL queries i.e attacker or hacker who wants to get 
access to backend database and information of organization/company creates 
malicious SQL queries that can easily get the information which are critical 
to organization/company.
2. Now let‘s understand How SQLI queries are created, how they are used to 
get information and types of sqli queries with a simple example.
3. Consider for example: You visit a website and it asks for login credentials to 
login i.e consider it asks for userid and password.
4. Consider you type the userid = ‗xyz‘and password = 123. The web 
application internally creates a sql query to validate the userid and password 
i.e sql query will be:
 Select * from users where userid = xyz and password = 123.
5. Consider users is the table name in database where they store userid and 
password for all users.
6. Once it matches the above userid and password. It will returns true that 
means login is successfull.
7. The attacker/hacker who does not has login credentials. Will Input the 
credentials like this.
 Userid = ‗or1=1--
 Password = 123.
8. Internally web application creates a sql query like this.
 Select * from user where userid = ‗or1=1— and password = 123
9. Userid contains or keyword. Or gate always returns true when one of the 
input is true. i.e 1=1 is always true. And – comments out rest of the query.
10. The result will be true. And attacker can get access to all userid and 
password in database.
11. This is the basic understanding of how sqli queries are used to get access to 
database. For more information.
 https://www.w3schools.com/sql/sql_injection.asp
Business Problem:
1. We are tasked with predicting the internally created sql queries by web 
application are SQLI queries or genuine SQL queries. Using Machine 
learning Techniques.
2. This can help the organization/company to better secure their backend 
database and customer information.
Machine Learning Formulation of business Problem:
1. It is a Binary classification problem. For a given SQl query. We need to 
predict the given query can get access to database or not.
Business Constraints:
 1. SQLI detection should not take more time. Some latency constraints are 
there.
 2. Misclassification of queries lead to breach of information of organization / 
company. So cost of misclassification is very high.
Dataset Column Analysis:
 1. The Dataset is taken from Kaggle:
 https://www.kaggle.com/sajid576/sql-injection-dataset
 2. Dataset contains two columns: Query, label
 3. Query column contains combination of SQLI queries, Genuine SQL queries
and plain text.
 4. Label column contains 1 and 0 values. Where 1 represents the Particular 
Query can get access to database i.e SQLI query and 0 represents can not get 
access to database i.e it can be Genuine Sql query or plain text.
 5. Number of rows in dataset is 30920.
 6. Dataset with label as 1 i.e SQLI queries:
 
7. Dataset with label 0 i.e Normal Queries or plain text :
Performance Metric:
This is basically a classification problem. Evaluation metrics used are:
 1. Accuracy.
 2. Confusion matrix.
 3. Recall rate
 4. Precision rate
 3. F1-Score
Research-Papers/Solutions/Architectures/Kernels
*** Mention the urls of existing research-papers/solutions/kernels on your problem statement and in your 
own words write a detailed summary for each one of them. If needed you can include images or explain 
with your own diagrams. it is mandatory to write a brief description about that paper. Without 
understanding of the resource please don‘t mention it***
1. https://hrcak.srce.hr/file/367636
a. Observation :
i. It is a multiclass classification problem. Normal SQL queries 
are labeled as 0, SQL Injection queries are labeled as 1 and 
Plain text are labeled as 2.
ii. Below is the table that describes the summary of distribution of 
different categories in dataset.
iii. Author mentioned Special characters, keywords, punctuations, 
Tags these are the features that differentiate between sql 
injection queries and normal sql queries or plain text. 
iv. Since the queries are combination of special characters, 
punctuation etc. Author has done lot of feature extraction.
v. Below table summarizes the feature extracted.
 
vi. Keywords plays major role in differentiating normal and 
malicious so to differentiate them Tokenization is performed. 
i.e Dividing each query into tokens.
vii. Each query is represented by continuous numbers where each 
number represents the above features.
viii. Author provides a below example to demonstrate how the 
Tokenization is done.
Consider the example or 1=1
After applying tokenization to above query. The output is 
ix. For training the models Author used various Ensemble models 
like Gradient Boosting Machine(GBM), Adaptive 
Boosting(AdaBoost), Extended Gradient Boosting 
Machine(XGBM) and Light Gradient boosting 
machine(LGBM).
x. Author trained the models with K-Fold cross validation mainly 
3 and 5 fold cross validation.
xi. Evaluation metric used are Accuracy,confusion matrix, 
precison,recall.roc curves.
b. Takeaways :
i. Most features are extracted based on special characters, 
Keywords, Punctuations present in queries field. Which are 
very much helpful for model to differentiate the normal and sqli 
queries. Feature extraction is most important process.
ii. We should come up with new features based on query field will 
be helpful.
iii. Among all Light GBM model gave the best accuracy of 99.3%.
2. https://www.irjet.net/archives/V8/i7/IRJET-V8I7515.pdf
a. Observations :
i. The paper focuses on both SQLI attacks and XSS attacks also 
called as Cross-Site scripting attacks.
ii. As a part of data preprocessing or data cleaning Author 
considers two approaches that are better.
a) Tokenization.
b) Removing stop words, stemming of words i.e NLP 
preprocessing techniques are used.
iii. The SQLI and XSS dataset is taken from kaggle.
https://www.kaggle.com/syedsaqlainhussain/sqlinjectiondataset?select=sqli.csv
https://www.kaggle.com/syedsaqlainhussain/crosssite-scripting-xssdataset-for-deep-learning
iv. Proposed System in the paper is shown below:
v. Author used Ensemble models for training : GBM, Adaboost, 
Xgboost, Light GBM.
vi. Evaluation metrics used are : Accuracy, Average Precision, 
Recall.
b. Takeaways :
i. Data Preprocessing or data cleaning plays a major important 
role in improving the model performance. Author used various 
data preprocessing techniques and improved the model 
performance.
ii. Among all the Ensemble models Light GBM provides more 
accuracy of 99.5%.
3. Malicious And Benign URLs | Kaggle
a. Observations :
i. This is not related to SQLI detection but aim of this problem is 
to detect malicious url.
ii. The Urls in the dataset are combination of special characters 
and keywords.
iii. The features extracted from the given urls will be helpful for 
our case study. 
iv. Features that are helpful :
1) Length of Query given.
2) Count Features : there are lot of special characters, 
keywords in query will take count of them for each query 
and perform EDA to test whether these features helpful 
in improving performance of model or not.
4. https://www.matecconferences.org/articles/matecconf/pdf/2018/32/matecconf_smima2018_010
04.pdf
a. Observations : 
i. This paper uses SVM Techniques to build the model.
ii. The dataset in this paper is does not contain sample queries and 
their label. It is Http request of websites.
iii. It contains lot of special symbols, characters etc. 
iv. This is how the sample data looks like.
Sql Injection sample:
page=2%25%27%20UNION%20ALL%20SELECT%20 
NULL%2C%20NULL%2C%20NULL%2C%20NULL 
%2C%20NULL%2C%20NULL%2C%20NULL%2C%2 
0NULL%2C%20NULL%2C%20NULL%2C%20NULL 
%2C%20NULL%2C%20NULL%2C%20NULL%2C%2 0NULL--%20
Normal Http request : 
action=http%3A//ytequocte.com/nhan-biet-dau-hieutrieu-chung-benhphukhoa/%3Fkeyword%3Dphu%2520khoa%26matchtype% 
3De%26adposition%3D1t4%26device%3Dc%26mtk%3 
Dsearch%26gclid%3DCLfa6O7Pj9MCFdAHKgodr5YO EA
v. Author used regular expression to take out only the important 
keywords and symbols from the dataset i.e by removing %, ?, 
space. After using regular expression the samples will be.
Positive sample: ['page=', '2', 'union', 'all', 'select', 'null', 'null', 'null', 'null', 
'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null', 'null']
Negative sample: ['action=', 'http://ytequocte.com', 'nhan', 'biet', 'dau', 
'hieu', 'trieu', 'chung', 'benh', 'phu', 'khoa', 'keyword=', 'phu', 'khoa', 
'matchtype=', 'e', 'adposition=', '1t4', 'device=', 'c', 'mtk=', 'search', 'gclid=', 
'clfa6o7pj9mcfdahkgodr5yoea']
vi. As a part of feature Extraction author performs word2vec to 
convert text features to n-dim vectors.
vii. Evaluation metrics used : Precision, Recall, Roc curve area.
b. Takeaways : 
i. Feature Extraction using word2vec preserves the semantic 
meaning between words which helps in improving the model 
performance.
ii. Svm algorithms performs well but not as compared to ensemble 
models from the above papers. 
5. Tokenization :
i. Almost all research papers above uses Tokenization method. Let‘s 
understand how this works.
ii. Tokenization is a Natural language preprocessing technique. Where 
we divide the given text into smaller token based on some delimeter.
iii. The delimeter can be space or any other character present in text.
iv. We can do word tokenize or character tokenize. i.e we can split the 
sentence to words or characters based on requirements.
v. We can take occurrences of word or character as a feature or feed as a 
vector to ML model.
vi. More about the tokenizer:
 https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenizationnlp/#:~:text=Tokenization%20is%20a%20way%20of,n%2Dgram%20charac
ters)%20tokenization
6. Ensemble Learning and Light GBM 
i. From above research papers we can infer that Ensemble models yield 
good accuracy. Specially Light GBM.
ii. In Ensemble learning multiple machine learning models are used 
together to make powerful model.
iii. The more different these machine learning models the better you can 
combine them and will get good results.
iv. There are four types Ensembles:
1.Bagging
2. Boosting
3.Stacking.
4.cascading.
v. From above research papers we can infer that Light GBM has shown 
the better performance and good accuracy.
vi. Xgboost and Light GBM are the implementation of gradient boosted 
machines
.
vii. Compared to Xgboost light GBM takes less time to train, and it is fast 
and gives high performance.
viii. More about Ensemble learning and Light GBM:
http://www.scholarpedia.org/article/Ensemble_learning
https://www.analyticsvidhya.com/blog/2017/06/which-algorithmtakes-the-crown-light-gbm-vs-xgboost/
First Cut Approach
*** Explain in steps about how you want to approach this problem and the initial experiments that you 
want to do. (MINIMUM 200 words) ***
*** When you are doing the basic EDA and building the First Cut Approach you should not refer any 
blogs or papers ***
The Approach will be based on the research and readings I have done.
1. From the above research data preprocessing, Feature Extraction are the 
important things to consider and they help in improving the model 
performance.
2. As a part of data preprocessing since the SQLI queries are composition of 
special characters, symbols etc. that makes them differentiate from normal 
text so we should not remove them, we can perform stemming, remove stop 
words.
3. As a part of featurization. tokenizing each query should be done. We can try 
out many possible tokenization methods and pick the best one. Some of 
them are listed below.
 1. Bag of words with unigram,bigram i.e n_gram range.
 2. Tf-idf with n_gram range
 3. Word2vec i.e avgword2vec and tfidf-word2vec
4. We can try out all the features mentioned in the above research papers and 
will find out how much of them are important or how much they are 
contributing to predicting target variable. By building simple model keeping 
that feature and target variable.
5. From the above research papers Ensemble models shown good results but 
will try out all the algorithms and choose the best model based on metrics 
used.
6. From the research papers mentioned above the metrics that can be used are : 
Accuracy, Precision, recall,F1-Score, Confusion_matrix.
7. From deep learning perspective we can pass the preprocessed data to bert 
model i.e by adding [CLS], [SEP], [PAD] tokens to text. Will take only the 
[CLS] token output i.e 786 dim vector and build a classifier on top of it.
Notes when you build your final notebook:
1. You should not train any model either it can be a ML model or DL model or Countvectorizer or 
even simple StandardScalar
2. You should not read train data files
3. The function1 takes only one argument ―X‖ (a single data points i.e 1*d feature) and the inside 
the function you will preprocess data point similar to the process you did while you featurize your 
train data
a. Ex: consider you are doing taxi demand prediction case study (problem definition: given 
a time and location predict the number of pickups that can happen)
b. so in your final notebook, you need to pass only those two values
c. def final(X):
preprocess data i.e data cleaning, filling missing values etc
compute features based on this X
use pre trained model
return predicted outputs
final([time, location])
d. in the instructions, we have mentioned two functions one with original values and one 
without it
e. final([time, location]) # in this function you need to return the predictions, no need to
compute the metric
f. final(set of [time, location] values, corresponding Y values) # when you pass the Y 
values, we can compute the error metric(Y, y_predict)
4. After you have preprocessed the data point you will featurize it, with the help of trained 
vectorizers or methods you have followed for your train data
5. Assume this function is like you are productionizing the best model you have built, you need to 
measure the time for predicting and report the time. Make sure you keep the time as low as 
possible
6. Check this live session: https://www.appliedaicourse.com/lecture/11/applied-machine-learningonline-course/4148/hands-on-live-session-deploy-an-ml-model-using-apis-on-aws/5/module-5-
feature-engineering-productionization-and-deployment-of-ml-models
