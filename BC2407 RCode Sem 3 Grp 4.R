setwd('D:/OY/NTU/Y2 Sem 2/BC2407 Course Materials/BC2407 Course Materials/Project')
Train <- read.csv('train_smote.csv')
Test <- read.csv('test.csv')

library(rpart)
library(rpart.plot) 
library(randomForest)
library(ggplot2)
library(caret) # for model evaluation
library(nnet)
library(readr)
library(neuralnet)
library(fastDummies)


set.seed(2024) # for reproducibility

str(Train)
str(Test)

# Change all data types to factor
Train$gender = factor(Train$gender)
Train$hypertension = factor(Train$hypertension)
Train$heart_disease = factor(Train$heart_disease)
Train$ever_married = factor(Train$ever_married)
Train$work_type = factor(Train$work_type)
Train$Residence_type = factor(Train$Residence_type)
Train$smoking_status = factor(Train$smoking_status)
Train$stroke = factor(Train$stroke)

Test$gender = factor(Test$gender)
Test$hypertension = factor(Test$hypertension)
Test$heart_disease = factor(Test$heart_disease)
Test$ever_married = factor(Test$ever_married)
Test$work_type = factor(Test$work_type)
Test$Residence_type = factor(Test$Residence_type)
Test$smoking_status = factor(Test$smoking_status)
Test$stroke = factor(Test$stroke)


#LOGISTIC REGRESSION----------------------------------------------------------------

levels(Train$stroke) # "0", "1"

# Including all variables
lm1 <- glm(stroke ~ ., family = binomial, data = Train)
summary(lm1)   #AIC: 5818.5
# bmi is insignificant

coef(lm1)
OR <- exp(coef(lm1))
OR

OR.CI <- exp(confint(lm1))
OR.CI
# at 95% confidence interval, bmi, smoking status CI includes 1, hence suggesting 
# that the variables are insignificant

# Remove bmi and smoking_status
lm2 <- glm(stroke ~ .-bmi-smoking_status, family = binomial, data = Train)
summary(lm2)    #AIC: 5982.6

prob <- predict(lm1, type = 'response')

# Set the threshold for predicting Y = 1 based on probability.
threshold <- 0.5

# If probability > threshold, then predict Y = 1, else predict Y = 0.
y.hat <- ifelse(prob > threshold, 1, 0)

# Create a confusion matrix with actuals on rows and predictions on columns.
table(Train$stroke, y.hat, deparse.level = 2)

# Overall Accuracy for trainset
mean(y.hat == Train$stroke)   #0.8246842

prob <- predict(lm1, newdata = Test, type = 'response')
# Set the threshold for predicting Y = 1 based on probability.
threshold <- 0.5

# If probability > threshold, then predict Y = 1, else predict Y = 0.
y.hat <- ifelse(prob > threshold, 1, 0)

# Create a confusion matrix with actuals on rows and predictions on columns.
table(Test$stroke, y.hat, deparse.level = 2)

# Overall Accuracy for testset
mean(y.hat == Test$stroke) #0.7738654


#SCALING----------------------------------------------------------------------------

Train$age_1 <- (Train$age - min(Train$age))/(max(Train$age)-min(Train$age))
Train$avg_glucose_level_1 <- (Train$avg_glucose_level - min(Train$avg_glucose_level))/(max(Train$avg_glucose_level)-min(Train$avg_glucose_level))
Train$bmi_1 <- (Train$bmi - min(Train$bmi))/(max(Train$bmi)-min(Train$bmi))

Test$age_1 <- (Test$age - min(Test$age))/(max(Test$age)-min(Test$age))
Test$avg_glucose_level_1 <- (Test$avg_glucose_level - min(Test$avg_glucose_level))/(max(Test$avg_glucose_level)-min(Test$avg_glucose_level))
Test$bmi_1 <- (Test$bmi - min(Test$bmi))/(max(Test$bmi)-min(Test$bmi))


#CART-------------------------------------------------------------------------------

m1 <- rpart(stroke ~. , data = Train, method = 'class',
            control = rpart.control(minsplit = 2, cp = 0))

# plots the maximal tree and results.
rpart.plot(m1, nn= T, main = "Maximal Tree in healthcare-dataset-stroke-data.csv")

# prints the maximal tree m1 onto the console.
print(m1)

# prints out the pruning sequence and 10-fold CV errors, as a table.
printcp(m1)

# Display the pruning sequence and 10-fold CV errors, as a chart.
plotcp(m1, main = "healthcare-dataset-stroke-data.csv")

#Finding optimal size of tree
CVerror.cap <- m1$cptable[which.min(m1$cptable[,"xerror"]), "xerror"] + m1$cptable[which.min(m1$cptable[,"xerror"]), "xstd"]
CVerror.cap
a <- 1; b<- 4
while (m1$cptable[a,b] > CVerror.cap) {
  a <- a + 1
}
a #optimal size of tree = 36

cp_optimal = ifelse(a > 1, sqrt(m1$cptable[a,1] * m1$cptable[a-1,1]), 1)

# Prune to optimal tree
m2 <- prune(m1, cp = cp_optimal)
print(m2)

m2$variable.importance

rpart.plot(m2, nn = T, tweak = 2.1, main = "Optimal Tree")

printcp(m2)

#testing for accuracy of CART

# Make predictions on both the training and testing datasets
train_pred <- predict(m2, newdata = Train, type = 'class')
test_pred <- predict(m2, newdata = Test, type = 'class')

# Create confusion matrices for both sets of predictions
train_conf_matrix <- confusionMatrix(as.factor(train_pred), as.factor(Train$stroke))
test_conf_matrix <- confusionMatrix(as.factor(test_pred), as.factor(Test$stroke))

# Display the confusion matrices
print(train_conf_matrix)   #accuracy = 0.965
print(test_conf_matrix)    #accuracy = 0.8693





#RANDOM FOREST-----------------------------------------------------------------------

# Build the Random Forest model
rf_model <- randomForest(stroke ~ ., data = Train, type = classification , ntree = 500, mtry = sqrt(ncol(Train) - 1), importance = TRUE)

# Print the model summary to see the results on training data
print(rf_model)
plot(rf_model)

#Variable Importance
importances <- importance(rf_model)
print(importances)
varImpPlot(rf_model)


# Use the trained model to make predictions on the test data
predictions <- predict(rf_model, Test, type= "response" )
# Convert predictions to a binary factor based on the chosen threshold (e.g., 0.5)
predicted_class <- ifelse(predictions == 2, 1, 0)
predicted_class <- as.factor(predicted_class)

#Error rate
err.rate <-rf_model$err.rate
View(err.rate)

# Make sure the actuals are also a factor with the same levels
actual_class <- factor(Test$stroke, levels = c(0, 1))

# Evaluate the model's performance
confusionMatrix <- table(Predicted = predictions, Actual = actual_class)
print(confusionMatrix)

accuracy <- sum(diag(confusionMatrix))/sum(confusionMatrix)
print(paste("Accuracy:", accuracy)) #90.1%



#NEURAL NETWORK------------------------------------------------------------------------

# Fit the neural network model
nn_model1 <- nnet(stroke ~ ., data = Train, size = 10, decay = 1e-4, maxit = 3000, linout = FALSE)

nn_model1

# Predict probabilities for the positive class (stroke)
predicted_probs <- predict(nn_model1, newdata=Test[, -which(names(Test) == "stroke")], type="raw")

# Determine class predictions based on a threshold (e.g., 0.5)
predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)

# Convert predicted_class to a factor for the confusion matrix
predicted_class <- factor(predicted_class, levels = c(0, 1))

# Ensure the actual stroke data is a factor with the same levels for comparison
actual_class <- factor(Test$stroke, levels = c(0, 1))

# Evaluate model performance
conf_matrix <- confusionMatrix(predicted_class, actual_class)
print(conf_matrix)

accuracy <- conf_matrix$overall['Accuracy']
print(paste("Accuracy:", round(accuracy * 100, 2), "%")) #78.79%



#model 2

nn_model2 <- nnet(stroke ~ ., data = Train, size = 40, decay = 1e-4, maxit = 3000, linout = FALSE)


# Predict probabilities for the positive class (stroke)
predicted_probs2 <- predict(nn_model2, newdata=Test[, -which(names(Test) == "stroke")], type="raw")

# Determine class predictions based on a threshold (e.g., 0.5)
predicted_class2 <- ifelse(predicted_probs2 > 0.5, 1, 0)

# Convert predicted_class to a factor for the confusion matrix
predicted_class2 <- factor(predicted_class2, levels = c(0, 1))

# Ensure the actual stroke data is a factor with the same levels for comparison
actual_class2 <- factor(Test$stroke, levels = c(0, 1))

# Evaluate model performance
conf_matrix2 <- confusionMatrix(predicted_class2, actual_class2)
print(conf_matrix2)

accuracy2 <- conf_matrix2$overall['Accuracy']
print(paste("Accuracy:", round(accuracy2 * 100, 2), "%")) #81.38%





#NEURAL NETWORK 2------------------------------------------------------------------

#model 1
#Dummy Variable

dcol <- c("work_type", "smoking_status")

Train <- dummy_cols(Train, remove_first_dummy = T, select_columns = dcol)

Test <- dummy_cols(Test, remove_first_dummy = T, select_columns = dcol)

Train$gender1 <- as.numeric(Train$gender)
Test$gender1 <- as.numeric(Test$gender)
Train$ever_married1 <- as.numeric(Train$ever_married)
Test$ever_married1 <- as.numeric(Test$ever_married)
Train$Residence_type1 <- as.numeric(Train$Residence_type)
Test$Residence_type1 <- as.numeric(Test$Residence_type)
Train$hypertension1 <- as.numeric(Train$hypertension)
Test$hypertension1 <- as.numeric(Test$hypertension)
Train$heart_disease1 <- as.numeric(Train$heart_disease)
Test$heart_disease1 <- as.numeric(Test$heart_disease)

set.seed(2024)
nn.1 <- neuralnet( stroke ~ gender1 + age + age_1 + hypertension1 + heart_disease1 + ever_married1 + work_type_1 + work_type_2 + work_type_3 + work_type_4 + Residence_type1 + avg_glucose_level + avg_glucose_level_1 + bmi + bmi_1 + smoking_status_1 + smoking_status_2 + smoking_status_3 , data=Train, hidden=2, stepmax =1e+06 , err.fct="ce", linear.output=FALSE)

#nn.1 <- neuralnet( stroke ~ . , data=Train, hidden=2, stepmax =1e+06 , err.fct="ce", linear.output=FALSE)
nn.1

par(mfrow=c(1,1))
plot(nn.1)

nn.1$net.result  # predicted outputs. 
nn.1$result.matrix  # summary
nn.1$startweights  #
nn.1$weights
# The generalized weight is defined as the contribution of the ith input variable to the log-odds:
nn.1$generalized.weights
## Easier to view GW as plots instead

par(mfrow=c(3,1))
gwplot(nn.1,selected.covariate="gender1", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="hypertension1", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="heart_disease1", min=-2.5, max=5)

par(mfrow=c(3,1))
#gwplot(nn.1,selected.covariate="ever_married1", min=-2.5, max=5)
#gwplot(nn.1,selected.covariate="Residence_type1", min=-2.5, max=5)

par(mfrow=c(4,1))
#gwplot(nn.1,selected.covariate="work_type_1", min=-2.5, max=5)
#gwplot(nn.1,selected.covariate="work_type_2", min=-2.5, max=5)
#gwplot(nn.1,selected.covariate="work_type_3", min=-2.5, max=5)
#gwplot(nn.1,selected.covariate="work_type_4", min=-2.5, max=5)

par(mfrow=c(3,1))
gwplot(nn.1,selected.covariate="smoking_status_1", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="smoking_status_2", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="smoking_status_3", min=-2.5, max=5)

par(mfrow=c(3,2))
gwplot(nn.1,selected.covariate="age_1", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="avg_glucose_level_1", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="bmi_1", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="age", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="avg_glucose_level", min=-2.5, max=5)
gwplot(nn.1,selected.covariate="bmi", min=-2.5, max=5)


pred.nn.1 <- predict(nn.1, Test, type = "response")
pred.nn.1 <- pred.nn.1[, -1]
pred.nn.1.class <- ifelse(pred.nn.1 > 0.5, 1, 0)

cm.nn.1 <- confusionMatrix(as.factor(pred.nn.1.class), Test$stroke)
cm.nn.1

cm.nn.1$overall['Accuracy'] #44.83568%


#model 2
nn.2 <- neuralnet( stroke ~ gender1 + age_1 + hypertension1 + heart_disease1 + ever_married1 + work_type_1 + work_type_2 + work_type_3 + work_type_4 + Residence_type1 + avg_glucose_level_1 + bmi_1 + smoking_status_1 + smoking_status_2 + smoking_status_3 , data=Train, hidden=2, stepmax =1e+06 , err.fct="ce", linear.output=FALSE)


par(mfrow=c(1,1))
plot(nn.2)

nn.2$net.result  # predicted outputs. 
nn.2$result.matrix  # summary
nn.2$startweights  #
nn.2$weights


# The generalized weight is defined as the contribution of the ith input variable to the log-odds:
nn.2$generalized.weights
## Easier to view GW as plots instead

par(mfrow=c(3,1))
gwplot(nn.2,selected.covariate="gender1", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="hypertension1", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="heart_disease1", min=-2.5, max=5,selected.response = 2)

par(mfrow=c(2,1))
gwplot(nn.2,selected.covariate="ever_married1", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="Residence_type1", min=-2.5, max=5,selected.response = 2)

par(mfrow=c(2,2))
gwplot(nn.2,selected.covariate="work_type_1", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="work_type_2", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="work_type_3", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="work_type_4", min=-2.5, max=5,selected.response = 2)

par(mfrow=c(3,1))
gwplot(nn.2,selected.covariate="smoking_status_1", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="smoking_status_2", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="smoking_status_3", min=-2.5, max=5,selected.response = 2)

par(mfrow=c(3,1))
gwplot(nn.2,selected.covariate="age_1", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="avg_glucose_level_1", min=-2.5, max=5,selected.response = 2)
gwplot(nn.2,selected.covariate="bmi_1", min=-2.5, max=5,selected.response = 2)


pred.nn.2 <- predict(nn.2, Test, type = "response")
pred.nn.2 <- pred.nn.2[, -1]
pred.nn.2.class <- ifelse(pred.nn.2 > 0.5, 1, 0)
cm.nn.2 <- confusionMatrix(as.factor(pred.nn.2.class), as.factor(Test$stroke))
cm.nn.2

#compare accuracy for both models

cm.nn.1$overall['Accuracy'] #44.83568%
cm.nn.2$overall['Accuracy'] #79.81221%


Test$prediction = pred.nn.2
write.csv(Test, file = "export.csv", row.names = T)
