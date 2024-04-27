# SVM and GBM
# Install and load required packages
install.packages(c("e1071", "gbm", "caret"))
library(e1071)
library(gbm)
library(caret)

# Load the datasets
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Explore the structure of the datasets
str(train)
str(test)

# Preprocess the data (handle missing values if any)
sum(is.na(train))
sum(is.na(test))

train[!complete.cases(train), ] <- lapply(train[!complete.cases(train), ], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
test[!complete.cases(test), ] <- lapply(test[!complete.cases(test), ], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Train the SVM model
svm_model <- svm(churn ~ ., data = train, kernel = "radial")

# Train the GBM model
gbm_model <- gbm(churn ~ ., data = train, distribution = "bernoulli", n.trees = 1000, verbose = FALSE)

# Make predictions on the test set using SVM
svm_predictions <- predict(svm_model, newdata = test)

# Make predictions on the test set using GBM
gbm_predictions <- predict.gbm(gbm_model, newdata = test, type = "response")

# Combine predictions by averaging
blended_predictions <- (svm_predictions + gbm_predictions) / 2

# Create a submission file
submission <- data.frame(id = test$id, churn = blended_predictions)
write.csv(submission, "submission.csv", row.names = FALSE)