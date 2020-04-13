#### Final Project Code


# load packages
library(dplyr)
library(caret)
library(randomForest)
library(rattle)
library(parallel)
library(doParallel)
library(ggplot2)

# set seed for replicability
set.seed(1234567)



## Basic Data Load and Cleaning

# read in data from csv files
training_full <- read.csv("train.csv")
validation <- read.csv("test.csv")



## Initial cleaning for training data set
# get data headers for training
headers <- names(training_full)
data_types <- sapply(training_full, class)

# convert true numeric data from factor
factors <- grep("factor", data_types)
factors <- factors[-c(1,2,3,37)]

asNumeric <- function(x) as.numeric(as.character(x))
training_full <- modifyList(training_full, lapply(training_full[, factors],   
                                                  asNumeric))

## Initial cleaning for validation data set
# get data headers for training
headers <- names(validation)
data_types <- sapply(validation, class)

# convert true numeric data from factor
factors <- grep("factor", data_types)
factors <- factors[-c(1,2,3,37)]

validation <- modifyList(validation, lapply(validation[, factors],   
                                            asNumeric))


## Initial Column Selection and Test/Train Data Creation

# identify the 8 caclucated features from data
feature_keys <- c("avg","var","stddev","max","min","amplitude","kurtosis","skewness")
toMatch <- paste0("^",feature_keys)
toMatch <- paste(toMatch, collapse = "|")
feature_names <- grep(toMatch, names(training_full), value = T)
feature_index <- grep(toMatch, names(training_full))

training_full_limited_cols <- training_full[,-c(feature_index)]
training_full_limited_cols <- training_full_limited_cols[,-c(1,3:7)]

# create testing/training samples within full testing data
inTraining <- createDataPartition(training_full_limited_cols$classe, p = .75, list=FALSE)
training <- training_full_limited_cols[inTraining,]
testing <- training_full_limited_cols[-inTraining,]

# save datasets for future use
saveRDS(training, file="training.RDS")
saveRDS(testing, file="testing.RDS")
saveRDS(validation, file="validation.RDS")




## Run Decision Tree and Random Forest Analysis

# configure training runs using various resampling methods
fitControl_none <- trainControl(method = "none",
                                allowParallel = TRUE)

fitControl_cv2 <- trainControl(method = "cv",
                               number = 2,
                               allowParallel = TRUE)

fitControl_cv5 <- trainControl(method = "cv",
                               number = 5,
                               allowParallel = TRUE)

fitControl_cv10 <- trainControl(method = "cv",
                                number = 10,
                                allowParallel = TRUE)

# configure parallel processing
cluster <- makeCluster(detectCores())
registerDoParallel(cluster)

# train tree models
start_tree_none <- Sys.time()
print(Sys.time())
modFit_tree_none <- train(classe ~ ., data=training, method="rpart", trControl=fitControl_none)
saveRDS(modFit_tree_none, "modFit_tree_none.RDS")
end_tree_none <- Sys.time()
print(Sys.time())

start_tree_cv2 <- Sys.time()
print(Sys.time())
modFit_tree_cv2 <- train(classe ~ ., data=training, method="rpart", trControl=fitControl_cv2)
saveRDS(modFit_tree_cv2, "modFit_tree_cv2.RDS")
end_tree_cv2 <- Sys.time()
print(Sys.time())

start_tree_cv5 <- Sys.time()
print(Sys.time())
modFit_tree_cv5 <- train(classe ~ ., data=training, method="rpart", trControl=fitControl_cv5)
saveRDS(modFit_tree_cv5, "modFit_tree_cv5.RDS")
end_tree_cv5 <- Sys.time()
print(Sys.time())

start_tree_cv10 <- Sys.time()
print(Sys.time())
modFit_tree_cv10 <- train(classe ~ ., data=training, method="rpart", trControl=fitControl_cv10)
saveRDS(modFit_tree_cv10, "modFit_tree_cv10.RDS")
end_tree_cv10 <- Sys.time()
print(Sys.time())


# train random forest models
start_rf_none <- Sys.time()
print(Sys.time())
modFit_rf_none <- train(classe ~ ., data=training, method="rf", trControl=fitControl_none)
saveRDS(modFit_rf_none, "modFit_rf_none.RDS")
end_rf_none <- Sys.time()
print(Sys.time())

start_rf_cv2 <- Sys.time()
print(Sys.time())
modFit_rf_cv2 <- train(classe ~ ., data=training, method="rf", trControl=fitControl_cv2)
saveRDS(modFit_rf_cv2, "modFit_rf_cv2.RDS")
end_rf_cv2 <- Sys.time()
print(Sys.time())

start_rf_cv5 <- Sys.time()
print(Sys.time())
modFit_rf_cv5 <- train(classe ~ ., data=training, method="rf", trControl=fitControl_cv5)
saveRDS(modFit_rf_cv5, "modFit_rf_cv5.RDS")
end_rf_cv5 <- Sys.time()
print(Sys.time())

start_rf_cv10 <- Sys.time()
print(Sys.time())
modFit_rf_cv10 <- train(classe ~ ., data=training, method="rf", trControl=fitControl_cv10)
saveRDS(modFit_rf_cv10, "modFit_rf_cv10.RDS")
end_rf_cv10 <- Sys.time()
print(Sys.time())

# de-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()



## Test model accuracy and predict

# predict with tree models
pred_tree_none <- predict(modFit_tree_none, testing)
pred_tree_cv2 <- predict(modFit_tree_cv2, testing)
pred_tree_cv5 <- predict(modFit_tree_cv5, testing)
pred_tree_cv10 <- predict(modFit_tree_cv10, testing)

# predict with random forest models
pred_rf_none <- predict(modFit_rf_none, testing)
pred_rf_cv2 <- predict(modFit_rf_cv2, testing)
pred_rf_cv5 <- predict(modFit_rf_cv5, testing)
pred_rf_cv10 <- predict(modFit_rf_cv10, testing)


# cacluate prediction accuracy on testing data set
accuracy_tree_none <- sum(pred_tree_none == testing$classe) / nrow(testing)
accuracy_tree_cv2 <- sum(pred_tree_cv2 == testing$classe) / nrow(testing)
accuracy_tree_cv5 <- sum(pred_tree_cv5 == testing$classe) / nrow(testing)
accuracy_tree_cv10 <- sum(pred_tree_cv10 == testing$classe) / nrow(testing)

accuracy_rf_none <- sum(pred_rf_none == testing$classe) / nrow(testing)
accuracy_rf_cv2 <- sum(pred_rf_cv2 == testing$classe) / nrow(testing)
accuracy_rf_cv5 <- sum(pred_rf_cv5 == testing$classe) / nrow(testing)
accuracy_rf_cv10 <- sum(pred_rf_cv10 == testing$classe) / nrow(testing)

accuracy_nums <- c(accuracy_tree_none, accuracy_tree_cv2, accuracy_tree_cv5, accuracy_tree_cv10,
                   accuracy_rf_none, accuracy_rf_cv2, accuracy_rf_cv5, accuracy_rf_cv10)

run_names <- c("tree_none", "tree_cv2", "tree_cv5", "tree_cv10", "rf_none", "rf_cv2", "rf_cv5", "rf_cv10")

accuracy <- as.data.frame(cbind(run_names, accuracy_nums))
accuracy$model <- c("tree", "tree", "tree", "tree", "rf", "rf", "rf", "rf")
names(accuracy) <- c("run", "accuracy", "model")
accuracy$accuracy <- as.numeric(as.character(accuracy$accuracy))

#g <- ggplot(data=accuracy, aes(x=run, y=accuracy, fill=model)) + geom_bar(stat="identity") + geom_text(aes(label=round(accuracy,4)), angle = 90, position = position_stack(vjust = 0.5))
#g + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + scale_x_discrete(limits=c(run_names)) + scale_y_continuous(labels = percent)


# Final Prediction Using Validation Data Set

# select relevant columns of the validation data set
final_cols <- names(training)
strMatch <- final_cols
strMatch <- paste0("^", strMatch, "$", collapse="|")
final_cols_index <- grep(strMatch, names(validation))
validation_final <- validation[,final_cols_index]

# perform prediction on the validation data set
pred_final <- predict(modFit_rf_cv5, validation_final)

