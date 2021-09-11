library(dplyr)
library(tidyverse)
library(gridExtra)
library(myEDA)

#Pull all the files for the project into an array
getwd()
files <- list.files(path = "Data/")

#Put extract training data
train_data <- read.csv(paste0("data/train.csv"), header = T, sep = ",")

#Modify Survived to a factor
train_data$Survived <- train_data$Survived %>% as.factor
train_data$Embarked <- as.factor(train_data$Embarked)
train_data$Cabin <- substr(train_data$Cabin, 0, 1)
train_data$Cabin <- as.factor(train_data$Cabin)

dir <- "Output/EDA/"

barchart_compare(train_data, 'Sex', 'Survived', dir, TRUE)
barchart_compare(train_data, 'Pclass', 'Survived', dir, TRUE)
barchart_compare(train_data, 'SibSp', 'Survived', dir, TRUE)
barchart_compare(train_data, 'Parch', 'Survived', dir, TRUE)
histogram_compare(train_data, 'Age', 'Survived', dir, TRUE)
histogram_compare(train_data, 'Fare', 'Survived', dir, TRUE)
barchart_compare(train_data, 'Cabin', 'Survived', dir, TRUE)
barchart_compare(train_data, 'Embarked', 'Survived', dir, TRUE)

