library(myEDA)
load.packages(c('dplyr',
                'plyr',
                'ISLR',
                'tree',
                'tidyverse',
                'gridExtra',
                'randomForest',
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

train_data$Survived <- as.factor(train_data$Survived)

#Impute the Age
#train_data$Age[is.na(train_data$Age)] <- median(train_data$Age, na.rm=TRUE)

#k-folds for testing
k <- 5
cv.mse_errors <- as.numeric(k)
cv.sae_errors <- as.numeric(k)
cv.mtry <- as.numeric(k)

#Fit to the regressor tree model
folds <- sample(1:5, nrow(train_data), replace = TRUE)


#Fit to the random tree model
set.seed(1)

for (i in 1:k) {
  start_time <- Sys.time()
  
  modellist <- list()
  
  control <- trainControl(method = "cv", number = 5, search = "grid")
  max_features <- ncol(train_data)-1
  tuneGrid <- expand.grid(.mtry = c(1:max_features))
  
  if (FALSE) {
    for (ntree in c(300, 500, 700, 900, 1100, 2000)) {
      set.seed(1)
      rForestTune.model <- train(Survived ~ ., 
                                 data = train_data[folds !=i,], 
                                 method = "rf", 
                                 metric = "Accuracy",
                                 tuneGrid = tuneGrid,
                                 trControl = control,
                                 ntree = ntree,
                                 na.action = na.roughfix)
      
      key <- toString(ntree)
      modellist[[key]] <- rForestTune.model
    }  
    
    # compare results
    results <- resamples(modellist)
    summary(results)
    dotplot(results)
  }|
  
  rForestTune.model <- train(Survived ~ ., 
                            data = train_data[folds !=i,], 
                            method = "rf", 
                            metric = "Accuracy",
                            tuneGrid = tuneGrid,
                            trControl = control,
                            na.action = na.roughfix,
                            ntree = 1000)

  bestMTRY <- rForestTune.model$bestTune$mtry

  rForest.model <- randomForest(formula = Survived ~ ., data = train_data[folds != i,], 
                                importance = FALSE, 
                                na.action = na.roughfix,
                                mtry = bestMTRY,
                                ntree = 1000)
  
  #Create predictions
  pred <- predict(rForest.model, newdata = train_data[folds==i,], type = "response")
  pred <- as.numeric(as.character(pred))
  
  #Female survive for NA predictions
  pred[is.na(pred)] <- ifelse(train_data$Sex[folds==i][is.na(pred)]=='female', 1, 0)
  
  cv.mse_errors[i] <- mean((as.numeric(as.character(train_data$Survived[folds==i])) - pred)^2)
  cv.sae_errors[i] <- sum(abs(as.numeric(as.character(train_data$Survived[folds==i])) - pred))
  cv.mtry[i] <- bestMTRY
  
  end_time <- Sys.time()
  print(end_time - start_time)
  invisible(gc())
}

cv.rForestTuned_mse <- mean(cv.mse_errors, na.rm = TRUE)
cv.rForestTuned_sae <- mean(cv.sae_errors, na.rm = TRUE)

#barchartPred_compare(train_data, 'Sex', 'Survived', 'yhat', dir, FALSE)
#barchartPred_compare(train_data, 'Pclass', 'Survived', 'yhat', dir, FALSE)
#histogramPred_compare(train_data, 'Age', 'Survived', 'yhat', dir, FALSE)

if (printSubmission) {
  test_data <- read.csv(paste0("data/test.csv"), header = T, sep = ",")
  test_data$Sex <- as.factor(test_data$Sex)
  test_data$Embarked <- as.factor(test_data$Embarked)
  test_data$Cabin <- substr(test_data$Cabin, 0, 1)
  test_data$Cabin <- as.factor(test_data$Cabin)
  
  #Impute the Age
  #test_data$Age[is.na(test_data$Age)] <- median(test_data$Age, na.rm=TRUE)
  
  #Impute the Fare
  #test_data$Fare[is.na(test_data$Fare)] <- median(test_data$Fare, na.rm=TRUE)
  
  #Fit to the train model
  control <- trainControl(method = "cv", number = 5, search = "grid")
  max_features <- ncol(train_data)-1
  
  tuneGrid <- expand.grid(.mtry = c(1:max_features))
  rForestTune.model <- train(Survived ~ ., 
                             data = train_data, 
                             method = "rf", 
                             metric = "Accuracy",
                             tuneGrid = tuneGrid,
                             trControl = control,
                             na.action = na.roughfix,
                             ntree = 1000)
  
  bestMTRY <- rForestTune.model$bestTune$mtry
  
  rForest.model <- randomForest(formula = Survived ~ ., data = train_data, 
                                importance = FALSE, 
                                na.action = na.roughfix,
                                mtry = bestMTRY,
                                ntree = 1000)
  
  
  #Make sure test data features can fit the training data
  levels(test_data$Sex) <- levels(train_data$Sex)
  levels(test_data$Embarked) <- levels(train_data$Embarked)
  levels(test_data$Cabin) <- levels(train_data$Cabin)
  
  test_data$Survived <- predict(rForest.model, newdata = test_data, type = "response")
  
  #For NA values use the female survive model
  test_data$Survived[is.na(test_data$Survived)] <- ifelse(test_data$Sex[is.na(test_data$Survived)]=='female', 1, 0)
  
  #Features to print in submission file
  submission_columns <- c('PassengerId', 'Survived')
  submission_data <- test_data[submission_columns]
  write.csv(submission_data, file = "output/Predictions/tunedRForestModel.csv", row.names = FALSE)
}
