# advanced stat learning assignment

library(splitTools) # used for splitting
library(modelr)     # used for cross validation
library(purrr)      # map function

directory = "C:\\Users\\micho\\OneDrive\\UNI\\HEC\\Winter Term 2022\\
Math80619A Advanced stat learning\\Assignment_H2022\\assignment_H2022\\"
file = "datrain.txt"
setwd(directory)
data = read.table(file,header = T, sep=" ",dec=".")
head(data)
typeof(data) # list
data = as.data.frame(data)
is.data.frame(data)
# transform y to a character
data$y = as.factor(data$y)
typeof(data$y)

# train validation split
inds = partition(data$fixedacidity, p = c(train = 0.75, valid = 0.25))

train = data[inds$train,]
valid = data[inds$valid,]

# function to calculate the accuracy of the model
accuracy = function(pred, real){
  return (100*sum(pred==real)/length(real))
}

#========================================================================
# NAIVE MODELS: random forest and SVM
# no order is considered, nominal classification
#   Random Forest 
library("randomForest") # Random Forest
# 4 folds cross validation function - return avg accuracy
run_4cv <- function(m, t, data){
  cv = crossv_kfold(data, k = 4)
  models = map(cv$train, ~randomForest(
    y~fixedacidity+volatileacidity+citricacid+residualsugar+chlorides+
      freesulfurdioxide+totalsulfurdioxide+density+pH+sulphates+alcohol,
    data = ., maxnodes = m, mtry = t))
  
  pred1 = predict(models$`1`,data[cv$test$`1`$idx,1:11],type = 'class')
  pred2 = predict(models$`2`,data[cv$test$`2`$idx,1:11],type = 'class')
  pred3 = predict(models$`3`,data[cv$test$`3`$idx,1:11],type = 'class')
  pred4 = predict(models$`4`,data[cv$test$`4`$idx,1:11],type = 'class')
  acc1 = accuracy(pred1, data[cv$test$`1`$idx,12])
  acc2 = accuracy(pred2, data[cv$test$`2`$idx,12])
  acc3 = accuracy(pred3, data[cv$test$`3`$idx,12])
  acc4 = accuracy(pred4, data[cv$test$`4`$idx,12])
  
  return(mean(c(acc1,acc2,acc3,acc4)))
}
# try different max nodes and number of variables values
mnodes = c(10,100,1000)
mtry = c(2,sqrt(11),4,5)
acc = c()
for (t in mtry){
  for (n in mnodes){
    acc_rf = run_4cv(m=n, t=t,data=data)
    acc = append(acc,acc_rf)
  }
}

for (t in mtry){
  acc_rf = run_4cv(m=NULL, t=t, data=data)
  acc = append(acc,acc_rf)
}

# print accuracy
print(paste("the best accuracy of the rf model is:",
            round(acc[which.max(acc)],digits = 3),"%")) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#   SVM
library("e1071")
run_4CVSVM <- function(g, c, data){
  cv = crossv_kfold(data, k = 4)
  models = map(cv$train, ~svm(
    y~fixedacidity+volatileacidity+citricacid+residualsugar+chlorides+
      freesulfurdioxide+totalsulfurdioxide+density+pH+sulphates+alcohol,
    data = ., gamma = g, cost = c))
  pred1 = predict(models$`1`,data[cv$test$`1`$idx,1:11],type = 'class')
  pred2 = predict(models$`2`,data[cv$test$`2`$idx,1:11],type = 'class')
  pred3 = predict(models$`3`,data[cv$test$`3`$idx,1:11],type = 'class')
  pred4 = predict(models$`4`,data[cv$test$`4`$idx,1:11],type = 'class')
  acc1 = accuracy(pred1, data[cv$test$`1`$idx,12])
  acc2 = accuracy(pred2, data[cv$test$`2`$idx,12])
  acc3 = accuracy(pred3, data[cv$test$`3`$idx,12])
  acc4 = accuracy(pred4, data[cv$test$`4`$idx,12])
  
  return(mean(c(acc1,acc2,acc3,acc4)))
}
acc_S = c()
for (g in c(0.1,0.5,1)){
  for (c in c(1,10,20,30)){
    acc_svm = run_4CVSVM(g=g, c=c, data=data)
    acc_S = append(acc_S,acc_svm)
  }
}

# print accuracy
print(paste("the best accuracy of the SVM model is:",
            round(acc_S[which.max(acc_S)],digits = 3),"%"))
#========================================================================
# Ordinal binary decomposition
# multiple model approach
library("rpart")
# prepare the data
train.v1 = train
train.v1$y = as.integer(train.v1$y)
train.v1$y[train.v1$y < 2] <- 0 # all the 1s are 0
train.v1$y[train.v1$y > 1] <- 1 # all the non 0s (previous 2s and 3s) are 1
train.v1$y = as.factor(train.v1$y)

train.v2 = train
train.v2$y = as.integer(train.v2$y)
train.v2$y[train.v2$y < 3] <- 0
train.v2$y[train.v2$y == 3] <- 1
train.v2$y = as.factor(train.v2$y)

# CV v1.rpart
run_cv1 <- function(maxd, minsplit, data){
  cv = crossv_kfold(data, k = 4)
  rpart.control(maxdepth = maxd, minsplit = minsplit)
  models = map(cv$train, ~rpart(
    y~fixedacidity+volatileacidity+citricacid+residualsugar+chlorides+
      freesulfurdioxide+totalsulfurdioxide+density+pH+sulphates+alcohol,
    data = .))
  pred1 = predict(models$`1`,data[cv$test$`1`$idx,1:11],type = 'class')
  pred2 = predict(models$`2`,data[cv$test$`2`$idx,1:11],type = 'class')
  pred3 = predict(models$`3`,data[cv$test$`3`$idx,1:11],type = 'class')
  pred4 = predict(models$`4`,data[cv$test$`4`$idx,1:11],type = 'class')
  acc1 = accuracy(pred1, data[cv$test$`1`$idx,12])
  acc2 = accuracy(pred2, data[cv$test$`2`$idx,12])
  acc3 = accuracy(pred3, data[cv$test$`3`$idx,12])
  acc4 = accuracy(pred4, data[cv$test$`4`$idx,12])
  return(mean(c(acc1,acc2,acc3,acc4)))
}
acc_r1 = c()
for(maxd in c(10,20,30)){
  for(mins in c(10,20,30)){
    acc_rp1 = run_cv1(maxd=maxd, minsplit=mins, data = train.v1)
    acc_r1 = append(acc_r1,acc_rp1)
  }
}

# CV v2.rpart
run_cv2 <- function(maxd, minsplit, data){
  cv = crossv_kfold(data, k = 4)
  rpart.control(maxdepth = maxd, minsplit = minsplit)
  models = map(cv$train, ~rpart(
    y~fixedacidity+volatileacidity+citricacid+residualsugar+chlorides+
      freesulfurdioxide+totalsulfurdioxide+density+pH+sulphates+alcohol,
    data = .))
  pred1 = predict(models$`1`,data[cv$test$`1`$idx,1:11],type = 'class')
  pred2 = predict(models$`2`,data[cv$test$`2`$idx,1:11],type = 'class')
  pred3 = predict(models$`3`,data[cv$test$`3`$idx,1:11],type = 'class')
  pred4 = predict(models$`4`,data[cv$test$`4`$idx,1:11],type = 'class')
  acc1 = accuracy(pred1, data[cv$test$`1`$idx,12])
  acc2 = accuracy(pred2, data[cv$test$`2`$idx,12])
  acc3 = accuracy(pred3, data[cv$test$`3`$idx,12])
  acc4 = accuracy(pred4, data[cv$test$`4`$idx,12])
  return(mean(c(acc1,acc2,acc3,acc4)))
}
acc_r2 = c()
for(maxd in c(10,20,30)){
  for(mins in c(10,20,30)){
    acc_rp2 = run_cv1(maxd=maxd, minsplit=mins, data = train.v2)
    acc_r2 = append(acc_r2,acc_rp2)
  }
}

rpart.control(minsplit = 10,maxdepth = 10)
v1.rpart <- rpart(formula = y~., 
                  data = train.v1, method = "class") # pr(target>1)
rpart.control(minsplit = 20,maxdepth = 20)
v2.rpart <- rpart(formula = y~., 
                  data = train.v2, method = "class") # pr(target>2)

pr.greater.v1 = predict(v1.rpart, valid[,1:11], type = 'prob')[,2]
pr.greater.v2 = predict(v2.rpart, valid[,1:11], type = 'prob')[,2]

pr.v1 = 1-pr.greater.v1                 # probability of class 1
pr.v2 = pr.greater.v1*(1-pr.greater.v2) # probability of class 2
pr.v3 = pr.greater.v2                   # probability of class 3

# get the index of the max probability
y_pred_mma = as.factor(t(apply(data.frame(pr.v1, pr.v2, pr.v3), 1, 
                               which.max)))

# evaluate the model
acc_mma = accuracy(y_pred_mma, valid$y)

# print accuracy
print(paste("the validation accuracy of the MMA model is:",
            round(acc_mma,digits = 3),"%"))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# multiple-output single model approach (NN) (ELM: Extreme Learning Machines)
library('elmNNRcpp')
# prepare the data
x.train = as.matrix(train[,1:11])
y.train = onehot_encode(as.numeric(train[,12])-1)
x.valid = as.matrix(valid[,1:11])

# train the model
elm1 = elm_train(x.train, y.train , nhid = 500, actfun = 'relu',
                 init_weights = "uniform_negative", 
                 bias = TRUE, verbose = TRUE)

# predict
y_pred_elm1 = elm_predict(elm1, x.valid, normalize = FALSE)
y_pred_elm1 = max.col(y_pred_elm1, ties.method = "random")

# evaluate the model
acc_elm1 = accuracy(y_pred_elm1, valid$y)

# print accuracy
print(paste("the validation accuracy of the ELM model is:"
            ,round(acc_elm1,digits = 3),"%"))

elm2 = elm_train(x.train, y.train , nhid = 1000, actfun = 'relu',
                 init_weights = "uniform_negative", bias = TRUE
                 , verbose = TRUE)

# predict
y_pred_elm2 = elm_predict(elm2, x.valid, normalize = FALSE)
y_pred_elm2 = max.col(y_pred_elm2, ties.method = "random")

# evaluate the model
acc_elm2 = accuracy(y_pred_elm2, valid$y)

# print accuracy
print(paste("the validation accuracy of the ELM model is:",
            round(acc_elm2,digits = 3),"%"))

elm3 = elm_train(x.train, y.train , nhid = 1500, actfun = 'relu',
                 init_weights = "uniform_negative", bias = TRUE, 
                 verbose = TRUE)

# predict
y_pred_elm3 = elm_predict(elm3, x.valid, normalize = FALSE)
y_pred_elm3 = max.col(y_pred_elm3, ties.method = "random")

# evaluate the model
acc_elm3 = accuracy(y_pred_elm3, valid$y)

# print accuracy
print(paste("the validation accuracy of the ELM model is:"
            ,round(acc_elm3,digits = 3),"%"))

#========================================================================
# threshold model
# proportional odds model
library('MASS')

run_cvp <- function(data){
  cv = crossv_kfold(data, k = 4)
  models = map(cv$train, ~polr(
    y~fixedacidity+volatileacidity+citricacid+residualsugar+chlorides+
      freesulfurdioxide+totalsulfurdioxide+density+pH+sulphates+alcohol,
    data = .))
  pred1 = predict(models$`1`,data[cv$test$`1`$idx,1:11],type = 'class')
  pred2 = predict(models$`2`,data[cv$test$`2`$idx,1:11],type = 'class')
  pred3 = predict(models$`3`,data[cv$test$`3`$idx,1:11],type = 'class')
  pred4 = predict(models$`4`,data[cv$test$`4`$idx,1:11],type = 'class')
  acc1 = accuracy(pred1, data[cv$test$`1`$idx,12])
  acc2 = accuracy(pred2, data[cv$test$`2`$idx,12])
  acc3 = accuracy(pred3, data[cv$test$`3`$idx,12])
  acc4 = accuracy(pred4, data[cv$test$`4`$idx,12])
  return(mean(c(acc1,acc2,acc3,acc4)))
}

print(paste("the validation accuracy of the POL model is:"
            ,round(run_cvp(data),digits = 3),"%"))

#========================================================================
# train best model on the whole data set
best_model = randomForest(y~., maxnodes = 100, mtry = 5, data=data)

# load test data
data_test = read.table('dateststudent.txt',header = T, sep=" ",dec=".")

pred = predict(best_model, data_test)
write.table(pred, file="prediction_Michel_Chamoun.txt",quote = F,
            col.names = F, row.names = F)
