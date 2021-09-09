library(dplyr)
library(ISLR)
library(tree)

#Pull all the files for the project into an array
getwd()
files <- list.files(path = "Data/")

#Put extract training data
train_data <- read.csv(paste0("data/train.csv"), header = T, sep = ",")
test_data <- read.csv(paste0("data/test.csv"), header = T, sep = ",")

#Reduce model to only features planned to be used
features <- c('Survived', 'Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked')
train_data <- train_data[, features]

#Convert character features to factor
train_data$Sex <- as.factor(train_data$Sex)
test_data$Sex <- as.factor(test_data$Sex)
train_data$Survived <- as.factor(train_data$Survived)

#Fit to the regressor tree model
set.seed(1)
tree.model <- tree(Survived ~ ., data = train_data)

#review the tree model
summary(tree.model)
plot(tree.model)
text(tree.model, pretty = 0, cex = 0.7)

#Check to see if pruning the tree would yield results
cv.tree.model <- cv.tree(tree.model)
plot(cv.tree.model$size, cv.tree.model$dev, type = 'b')

#plots demonstrate that 6 nodes produce the lowest cv error rate
prune.tree.model <- prune.tree(tree.model, best = 6)
plot(prune.tree.model)
text(prune.tree.model, pretty=0, cex = 0.7)

#make predictions on train_data
train_data$yhat <- predict(prune.tree.model, newdata = train_data, type = "class")

#plot predictions on actuals vs expected
table(train_data$yhat, train_data$Survived)

#make predictions on test_data
test_data$Survived <- predict(prune.tree.model, newdata = test_data, type = "class")

#Features to print in submission file
submission_columns <- c('PassengerId', 'Survived')
submission_data <- test_data[submission_columns]
write.csv(submission_data, file = "output/PrunedTreeModel.csv", row.names = FALSE)
