# SQL-Injection--Detection-using-Machine-Learning

The Aim of this case study is to build a robust machine learning model that can detect a sql injection queries which in turn better secure the organization / company from the attacker / hacker.

The more detailed demonstration and explaination on how the model is built using machine learning and deep learning. what are the features that are extracted. and what are the references or articles i have used. Everything I have written in the below blog.

Blog link : https://medium.com/@rohanhavannavar/2c9303024da3

Below is the link of web application which is hosted on AWS EC2 Instance.

Web application link : http://ec2-13-57-16-81.us-west-1.compute.amazonaws.com:8080/index

Below is the link for demonstration video of how the web application works and to check whether the machine learning model predicting it correctly or not.

Video link : https://www.youtube.com/watch?v=AuelUl7j47w

for the word2vec embedding we have used Pretrained glove vectors you can download it by visiting the link : https://nlp.stanford.edu/projects/glove/

I have explained how to deploy the application to localhost using flask API in the above given blog. 

Now let us look at how to deploy it in AWS EC2 instance.

First create AWS account and launch an EC2 instance from AWS management console.

Choose Ubuntu os free tier and in security groups set the inbound rule to port 8080 and launch the instance. 

login in remote box using the command prompt in windows and terminal in MAC os 

in command prompt enter the command
ssh -i "here you have to enter the name of key-value pair file given by AWS" "public server name given by AWS"

now from command prompt enter the below command to copy the flask file to remote box.
in the above code we have flask.zip download it this is the file we have to copy to remote box.
scp -r -i "here you have to enter the name of key-value pair file given by AWS" "path of the folder in local machine" "public server name given by AWS"

now in remote box change the directory to that folder and run the below command
python3 app.py
now you will see the web application running.

for more information how to deploy flask app on AWS follow the link : https://www.youtube.com/watch?v=_rwNTY5Mn40
