library(caret)

#Load dataset
data(iris)
dataset = iris

#Create a dataset using 80% of rows
validation_index = createDataPartition(dataset$Species, p=0.80, list=FALSE)

#Selet 20% of the dataset for validation
validation = dataset[-validation_index,]

#Use the remaining 80% of the dataset for training and testing
dataset = dataset[validation_index,]

#Show dimesions of dataset
dim(dataset)

#List types for each atribute
sapply(dataset, class)

#Show first 5 rows of dataset
head(dataset)

#List the levels for each class
levels(dataset$Species)

#Summarize class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# summarize attribute distributions
summary(dataset)

#Define x and y
x = dataset[,1:4]
y = dataset[,5]

#Create a boxplot for each attribute
par(mfrow=c(1,4))
  for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

#Create a barplot for class breakdown
plot(y)

#Create scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

#Create boxplots for each attribute
featurePlot(x=x, y=y, plot="box")

#Create density plots for each attribute by class value
scales = list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#Run algorithms using 10-fold cross validation
control = trainControl(method="cv", number=10)
metric = "Accuracy"

#Linear algorithms
set.seed(7)
fit.lda = train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

#Nonlinear algorithms
# CART
set.seed(7)
fit.cart = train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(7)
fit.knn = train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

#Advanced algorithms
# SVM
set.seed(7)
fit.svm = train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

# Random Forest
set.seed(7)
fit.rf = train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

#Summarize accuracy of models
results = resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

#Compare accuracy of models
dotplot(results)

#Summarize Best Model
print(fit.lda)

#Estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)


