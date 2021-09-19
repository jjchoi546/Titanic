library(dplyr)

#Print submission?
printSubmission <- FALSE

#Pull all the files for the project into an array
getwd()
files <- list.files(path = "Data/")

#Put extract training data
train_data <- read.csv(paste0("data/train.csv"), header = T, sep = ",")

#Check how many are male/female
survFemale <- nrow(train_data[(train_data$Sex=="female" & train_data$Survived==1),])
survMale <- nrow(train_data[(train_data$Sex=="male" & train_data$Survived==1),])
allFemale <- nrow(train_data[(train_data$Sex=="female"),])
allMale <- nrow(train_data[(train_data$Sex=="male"),])

rateFemale <- survFemale/allFemale
rateMale <- survMale/allMale

print(paste0("% of women who survived: ", round(rateFemale, 4)))
print(paste0("% of men who survived: ", round(rateMale, 4)))

#k-folds for all testing
k <- 5
cv.mse_errors <- as.numeric(k)
cv.sae_errors <- as.numeric(k)

folds <- sample(1:5, nrow(train_data), replace = TRUE)

for (i in 1:k) {
  start_time <- Sys.time()
  
  pred <- 0
  cv.mse_errors[i] <- mean((as.numeric(as.character(train_data$Survived[folds==i])) - pred) ^2)
  cv.sae_errors[i] <- sum(abs((as.numeric(as.character(train_data$Survived[folds==i])))))
  
  end_time <- Sys.time()
  print(end_time - start_time)
  invisible(gc())
}

cv.base_mse <- mean(cv.mse_errors, na.rm = TRUE)
cv.base_sae <- mean(cv.sae_errors, na.rm = TRUE)

if (printSubmission) {
  test_data <- read.csv(paste0("data/test.csv"), header = T, sep = ",")

  #Simple model. Survive when female, die when male.
  test_data <- test_data %>% mutate(Survived = case_when(Sex == "female" ~ 1, 
                                             Sex == "male" ~ 0))

  #Prepare submission files
  submission_columns <- c("PassengerId", "Survived")
  submission_data <- test_data[submission_columns]
  write.csv(submission_data, file = "Output/Predictions/FemaleSurv_Model.csv", row.names = FALSE)
}