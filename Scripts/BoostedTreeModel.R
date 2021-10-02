library(myEDA)
load.packages(c('dplyr',
                'plyr',
                'ISLR',
                'tree',
                'tidyverse',
                'gridExtra',
                'gbm',
                'caret'))

#Print Submission?
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

#Boosted model needs charactor predictor
train_data$Survived <- as.character(train_data$Survived)

#k-folds for testing
k <- 5
cv.mse_errors <- as.numeric(k)
cv.sae_errors <- as.numeric(k)

#Fit to the regressor tree model
folds <- sample(1:5, nrow(train_data), replace = TRUE)


#Fit to the bosted model with standard parameters
set.seed(1)

Boost.model <- gbm(formula = Survived ~ ., 
                   data = train_data,
                   distribution = "bernoulli",
                   n.trees = 10000,
                   interaction.depth = 1,
                   shrinkage = 0.001,
                   verbose = FALSE,
                   cv.folds = 5)

print(Boost.model)
sqrt(min(Boost.model$cv.error))
gbm.perf(Boost.model, method = 'cv')
invisible(gc())

set.seed(1)
Boost2.model <- gbm(formula = Survived ~ ., 
                   data = train_data,
                   distribution = "bernoulli",
                   n.trees = 10000,
                   interaction.depth = 3,
                   shrinkage = 0.001,
                   verbose = FALSE,
                   cv.folds = 5)

print(Boost2.model)
sqrt(min(Boost2.model$cv.error))
gbm.perf(Boost2.model, method = 'cv')
invisible(gc())

hyper_grid <- expand.grid(shrinkage = c(.0059, .006, .0061),
                         interaction.depth = c(3, 4, 5),
                         n.minobsinnode = c(4, 5, 6),
                         bag.fraction = 1,
                         optimal_trees = 0,
                         min_RMSE = 0)

for(i in 1:nrow(hyper_grid)) {
  set.seed(1)
  
  Boost.tune <- gbm(formula = Survived~.,
                    distribution = "bernoulli",
                    data = train_data,
                    n.trees = 5000,
                    interaction.depth = hyper_grid$interaction.depth[i],
                    shrinkage = hyper_grid$shrinkage[i],
                    n.minobsinnode = hyper_grid$n.minobsinnode[i],
                    bag.fraction = hyper_grid$bag.fraction[i],
                    train.fraction = .8,
                    n.cores = NULL,
                    verbose = FALSE)
  
  hyper_grid$optimal_trees[i] <- which.min(Boost.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(Boost.tune$valid.error))
  invisible(gc())
}

hyper_grid %>% dplyr::arrange(min_RMSE) %>% head(10)

set.seed(1)
for (i in 1:k) {
  start_time <- Sys.time()
  
  Boost.model <- gbm(formula = Survived ~ ., data = train_data[folds != i,], 
                     distribution = "bernoulli",
                     n.trees = 5000,
                     interaction.depth = 4)
  
  #Create predictions
  pred <- predict(Boost.model, 
                  newdata = train_data[folds == i,], 
                  n.trees=5000,
                  shrinkage = 0.006,
                  interaction.depth = 3,
                  n.minobsinnode = 5,
                  bag.fraction = 1,
                  type = "response")
  
  pred <- ifelse(pred > 0.5, 1, 0)

  cv.mse_errors[i] <- mean((as.numeric(as.character(train_data$Survived[folds==i])) - pred)^2)
  cv.sae_errors[i] <- sum(abs(as.numeric(as.character(train_data$Survived[folds==i])) - pred))
  
  end_time <- Sys.time()
  print(end_time - start_time)
  invisible(gc())
}

cv.Boost_mse <- mean(cv.mse_errors, na.rm = TRUE)
cv.Boost_sae <- mean(cv.sae_errors, na.rm = TRUE)


if (printSubmission) {
  test_data <- read.csv(paste0("data/test.csv"), header = T, sep = ",")
  
  #Convert character features to factor
  test_data$Sex <- as.factor(test_data$Sex)
  test_data$Embarked <- as.factor(test_data$Embarked)
  test_data$Cabin <- substr(test_data$Cabin, 0, 1)
  test_data$Cabin <- as.factor(test_data$Cabin)
  
  Boost.model <- gbm(formula = Survived ~ ., data = train_data, 
                     distribution = "bernoulli",
                     n.trees = 5000,
                     shrinkage = 0.006,
                     n.minobsinnode = 5,
                     interaction.depth = 3,
                     bag.fraction = 1,
                     type = "response")
  
  pred <- predict(Boost.model, 
                  newdata = test_data, 
                  n.trees=5000,
                  shrinkage = 0.006,
                  interaction.depth = 3,
                  n.minobsinnode = 5,
                  bag.fraction = 1,
                  type = "response")
  
  pred <- ifelse(pred > 0.5, 1, 0)
  
  test_data$Survived <- pred
  
  #Features to print in submission file
  submission_columns <- c('PassengerId', 'Survived')
  submission_data <- test_data[submission_columns]
  write.csv(submission_data, file = "output/Predictions/BoostModel.csv", row.names = FALSE)
}

