# Starbucks Capstone Challenge

### Introduction

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 

# Installation

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.

# Project Motivation

It is the Starbuck's Capstone Challenge of the Data Scientist Nanodegree in Udacity. We get the dataset from the program that creates the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers. 

I'm interested to answer the following questions:

1. What is the majority gender of the customers and what’s their average income ?
2. What is the correlation between the age and the average income?
3. what is the correlation between (gender and age), and (gender and income) ?
4. Would we be able to classify customers based on their income ?


# Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


# Results
1. Among the different transcript events, the offers were altered into transactions, however less than 25% of those completed the offer.
 
2. Most of the users are males
 
3. The overall average income is higher in female than male.
 
4. Customers 49 years old and older gain more that the overall average income.
 
5. The year 2018 had more registered customers in compare to 2014-2016
 
6. The age of the registered female customers is higher than the male customers.
 
7.  I have tried 2 machine learning algorithms; Decision Tree and Naive Bayes, both had 57% accuracy rate. Tried to tune parameters, however result is still the same

8. In future I would try the classification models on larger dataset


### An article that summerize my findings can be found here:
https://medium.com/@malshedokhi/data-science-starbucks-capstone-challenge-6b567e6f8374
