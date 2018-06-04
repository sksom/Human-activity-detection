# human activity recognition
# Deep learning using keras
library(keras)

# import data
data1 <- read.csv(file.choose(),header = T) # train
data2 <- read.csv(file.choose(),header = T) # test
str(data1)
str(data2)

# PCA bcz there are too many variables to train
#pc1 <- prcomp(data1[,-563], center = T, scale. = T)
#pc2 <- prcomp(data2[,-563], center = T, scale. = T)
#summary(pc1)
#summary(pc2)

# conversion of data to PCAed data
#data <- predict(pc1,data1)
data <- data1
#data <- data.frame(data, data1[563])

#tstdata <- predict(pc2,data2)
tstdata <- data2
#tstdata <- data.frame(tstdata, data2[563])

# change data to matrix
data <- as.matrix(data)
dimnames(data) <- NULL
str(data) # num [1:7352, 1:563]

tstdata <- as.matrix(tstdata)
dimnames(tstdata) <- NULL
str(tstdata) # num [1:2947, 1:563]

# Normalize data except the response variable
# data[,1:562] <- normalize(data[,1:562])
data[,563] <- as.numeric(data1[,563])-1
tstdata[,563] <- as.numeric(data2[,563])-1
summary(data)

summary(tstdata)

# data partition
#set.seed(1234)
#ind <- sample(2,nrow(data),replace = T,prob = c(0.7,0.3))
#training <- data[ind==1,1:21]
#test <- data[ind==2,1:21]
training <- data[,1:562]  # 150 and not 562 bcz we did PCA
test <- tstdata[,1:562]
trainingTarget <- data[,563]
testTarget <- tstdata[,563]

# one-hot encoding
trainLabels <- to_categorical(trainingTarget)
testLabels <- to_categorical(testTarget)
print(testLabels)

# model, units=8(experimantal), input_shape= 21, bcz 21 input variables, next units=3
# bcz three response classes 0,1,2
model <- keras_model_sequential()
model %>%
  layer_dense(units = 8, activation = "relu",input_shape = c(562)) %>%
  layer_dense(units = 6, activation = "softmax")
summary(model)

# compile, for only two response classes use "binary_crossentropy"
model %>%
  compile(loss = "categorical_crossentropy", optimizer = "adam", 
          metrics = "accuracy")

# fit the model
history1 <- model %>%
  fit(training,trainLabels,epochs = 100,batch_size = 32,validation_split = 0.2)
plot(history1)

# evaluate model for test data
model %>% evaluate(test,testLabels)

# Prediction and confusion matrix - test data
pred <- model %>%
  predict_classes(test)
prob <- model %>%
  predict_proba(test)
table(Predicted=pred,Actual=testTarget)
cbind(prob,Predicted=pred,Actual=testTarget)

# acc= 0.9470648
