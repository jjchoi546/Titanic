library(dplyr)
library(ISLR)
library(tree)
library(tidyverse)
library(gridExtra)
library(myEDA)

#Print submission?
printSubmission <- FALSE

#Pull all the files for the project into an array
getwd()
files <- list.files(path = "Data/")

#Put extract training data
train_data <- read.csv(paste0("data/train.csv"), header = T, sep = ",")

#Reduce model to only features planned to be used
features <- c('Survived', 'Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin')
train_data <- train_data[, features]

#Convert character features to factor
train_data$Sex <- as.factor(train_data$Sex)

train_data$Embarked <- as.factor(train_data$Embarked)

train_data$Cabin <- substr(train_data$Cabin, 0, 1)
train_data$Cabin <- as.factor(train_data$Cabin)

train_data$Survived <- as.factor(train_data$Survived)

#k-folds for testing
k <- 5
cv.mse_errors <- as.numeric(k)
cv.sae_errors <- as.numeric(k)

#Fit to the regressor tree model
folds <- sample(1:5, nrow(train_data), replace = TRUE)

set.seed(1)
for (i in 1:k) {
  start_time <- Sys.time()
  
  tree.model <- tree(Survived ~ ., data = train_data[folds != i,])
  
  #review the tree model
  #summary(tree.model)
  #plot(tree.model)
  #text(tree.model, pretty = 0, cex = 0.7)
  
  #Check to see if pruning the tree would yield results
  cv.tree.model <- cv.tree(tree.model)
  #plot(cv.tree_model$size, cv.treemodel$dev, type = 'b')
  min.idx <- which.min(cv.tree.model$dev)
  min.node <- cv.tree.model$size[min.idx]
  
  prune.tree.model <- prune.tree(tree.model, best = min.node)
  #plot(prune.tree.model)
  #text(prune.tree.model, pretty=0, cex=0.7)
  
  #make predictions on train_data
  pred <- predict(prune.tree.model, newdata = train_data[folds==i,], type = "class")
  pred <- as.numeric(as.character(pred))
  
  #calculate errors
  cv.mse_errors[i] <- mean((as.numeric(as.character(train_data$Survived[folds==i])) - pred)^2)
  cv.sae_errors[i] <- sum(abs(as.numeric(as.character(train_data$Survived[folds==i])) - pred))
  
  end_time <- Sys.time()
  print(end_time - start_time)
  invisible(gc())
}

cv.pTree_mse <- mean(cv.mse_errors, na.rm = TRUE)
cv.pTree_sae <- mean(cv.sae_errors, na.rm = TRUE)



#plot predictions on actuals vs expected
#table(train_data$yhat, train_data$Survived)

#barchartPred_compare(train_data, 'Sex', 'Survived', 'yhat', dir, FALSE)
#barchartPred_compare(train_data, 'Pclass', 'Survived', 'yhat', dir, FALSE)
#histogramPred_compare(train_data, 'Age', 'Survived', 'yhat', dir, FALSE)

if (printSubmission) {
  test_data <- read.csv(paste0("data/test.csv"), header = T, sep = ",")
  
  
  #make predictions on test_data
  test_data$Survived <- predict(prune.tree.model, newdata = test_data, type = "class")
  
  test_data$Sex <- as.factor(test_data$Sex)
  test_data$Embarked <- as.factor(test_data$Embarked)
  test_data$Cabin <- substr(test_data$Cabin, 0, 1)
  test_data$Cabin <- as.factor(test_data$Cabin)

  #Features to print in submission file
  submission_columns <- c('PassengerId', 'Survived')
  submission_data <- test_data[submission_columns]
  write.csv(submission_data, file = "output/Predictions/PrunedTreeModel.csv", row.names = FALSE)
}