# Diabetes project
# Wenying Quan

# Install required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot")
if(!require(dplyr)) install.packages("dplyr")

library(tidyverse)
library(caret)
library(corrplot)
library(dplyr)

# Pima Indians diabetes dataset
# https://www.kaggle.com/uciml/pima-indians-diabetes-database/version/1

# Importing data
diab <- read.csv("Diabetes.csv",header = T)
nrow(diab)
head(diab)
ggplot(diab, aes(x=Outcome)) + geom_bar(fill="black",alpha=0.7) + 
  labs(title = "Distribution of Diagnosis")

##########################################################
# Univariate analysis
##########################################################
# 1. Number of pregnancies
summary(diab$Pregnancies)
hist(diab$Pregnancies,main = "Histogram of Pregnancies", xlab = "Pregnancies")

# 2. Glucose
summary(diab$Glucose)
hist(diab$Glucose,main = "Histogram of Glucose", xlab = "Glucose")
# Substitute missing values with mean of Glucose.
diab$Glucose <- na_if(diab$Glucose,0)
diab$Glucose[is.na(diab$Glucose)] <- mean(diab$Glucose, na.rm = T)
summary(diab$Glucose)
hist(diab$Glucose,main = "Histogram of Glucose", xlab = "Glucose")

# 3. Blood Pressure
summary(diab$BloodPressure)
hist(diab$BloodPressure,main = "Histogram of Blood Pressure", xlab = "Blood Pressure")
# Substitute missing values with mean of Blood Pressure.
diab$BloodPressure <- na_if(diab$BloodPressure,0)
diab$BloodPressure[is.na(diab$BloodPressure)] <- mean(diab$BloodPressure, na.rm = T)
summary(diab$BloodPressure)
hist(diab$BloodPressure,main = "Histogram of Blood Pressure", xlab = "Blood Pressure")

# 4. Skin Thickness
summary(diab$SkinThickness)
hist(diab$SkinThickness,main = "Histogram of Skin Thickness", xlab = "Skin Thickness")
# Substitute missing values with median of Skin Thickness.
diab$SkinThickness <- na_if(diab$SkinThickness,0)
diab$SkinThickness[is.na(diab$SkinThickness)] <- median(diab$SkinThickness, na.rm = T)
summary(diab$SkinThickness)
hist(diab$SkinThickness,main = "Histogram of Skin Thickness", xlab = "Skin Thickness")

# 5. Insulin
summary(diab$Insulin)
hist(diab$Insulin,main = "Histogram of Insulin", xlab = "Insulin")
# Substitute missing values with median of Insulin.
diab$Insulin <- na_if(diab$Insulin,0)
diab$Insulin[is.na(diab$Insulin)] <- median(diab$Insulin, na.rm = T)
summary(diab$Insulin)
hist(diab$Insulin,main = "Histogram of Insulin", xlab = "Insulin")

# 6. BMI
summary(diab$BMI)
hist(diab$BMI,main = "Histogram of BMI", xlab = "BMI")
# Substitute missing values with median of BMI
diab$BMI <- na_if(diab$BMI,0)
diab$BMI[is.na(diab$BMI)] <- median(diab$BMI, na.rm = T)
summary(diab$BMI)
hist(diab$BMI,main = "Histogram of BMI", xlab = "BMI")

# 7. Diabetes Pedigree Function
summary(diab$DiabetesPedigreeFunction)
hist(diab$DiabetesPedigreeFunction,main = "Histogram of Diabetes Pedigree Function", 
     xlab = "Diabetes Pedigree Function")

# 8. Age
summary(diab$Age)
hist(diab$Age,main = "Histogram of Age", xlab = "Age")

# Correlation check between 8 risk factors.
corrplot(cor(diab[1:8]),tl.col = "black")
correlation <- cor(diab[1:8])
findCorrelation(correlation, cutoff = 0.7)


##########################################################
# Splitting the data into training and test data
##########################################################
set.seed(1,sample.kind="Rounding")
test_index <-createDataPartition(y = diab$Outcome, times = 1, p = 0.25, list = FALSE)
train <-diab[-test_index,]
test <- diab[test_index,]


##########################################################
# Model Fitting
##########################################################
# 1. Logistic Regression Model
fit_glm <- glm(Outcome ~., data=train, family="binomial")
p_hat_glm <- predict(fit_glm, test)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5,1,0))
glm_confusion <- confusionMatrix(data = y_hat_glm, reference = factor(test$Outcome))
glm_confusion
varImp(fit_glm)

# 2. K Nearest Neighbor(KNN) Model
train$Outcome <- factor(train$Outcome)
test$Outcome <- factor(test$Outcome)
train_knn<- train(Outcome~., method = "knn", data = train)
y_hat_knn <- predict(train_knn,test)
knn_confusion <- confusionMatrix(data = y_hat_knn, reference=test$Outcome)
knn_confusion
varImp(train_knn)

# 3. Random Forest
train$Outcome <- factor(train$Outcome)
test$Outcome <- factor(test$Outcome)
train_rf <- train(Outcome~., method="rf", data=train)
rf_confusion <- confusionMatrix(predict(train_rf,test),test$Outcome)
rf_confusion
varImp(train_rf)


##########################################################
# Final result
##########################################################
confusionmatrix <- list(logistic_regression = glm_confusion,
                        knn = knn_confusion, random_forest = rf_confusion)
final_result <- sapply(confusionmatrix, function(x) x$byClass)           
final_result %>% knitr::kable()
