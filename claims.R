setwd("~/projects/datascience/claims_severity")
data <- read.csv('data/train.csv', quote="\"", comment.char="", stringsAsFactors = FALSE)
test.data <- read.csv('data/test.csv', quote="\"", comment.char="", stringsAsFactors = FALSE)

################## Pre-process data ################################
trainrows <- read.csv("data/trainrows09.csv")[,1] #sample(1:nrow(df),nrow(df)*0.6)
valrows <- read.csv("data/crossrows09.csv")[,1] #(which(!1:nrow(df) %in% c(trainrows, testrows))) write.csv(file='testrows.csv', testrows, row.names = FALSE)

df <- data.frame(data)
test.df <- data.frame(test.data)

test.rows.diff.factor <- c()
cols.diff.factor <- c()
for (i in 1:ncol(df)) {
  if (startsWith(colnames(df)[i], 'cat')) {
    df[,i] <- as.factor(df[,i])
    test.df[,i] <- as.factor(test.df[,i])
    if (!all(levels(test.df[,i]) %in% levels(df[,i]))) {
      cols.diff.factor <- c(cols.diff.factor, colnames(df)[i])
      test.rows.diff.factor <- c(test.rows.diff.factor, which(test.df[,i] %in% setdiff(levels(test.df[,i]), levels(df[,i]))))
    }
  }
  if (startsWith(colnames(df)[i], 'cont')) {
    df[,i] <- scale(df[,i])
    test.df[,i] <- scale(test.df[,i])
  }
}
test.rows.diff.factor <- unique(test.rows.diff.factor)
cols.diff.factor <- colnames(df[,setdiff(1:ncol(df), which(colnames(df) %in% c("id","loss", "logloss", cols.diff.factor)))])

output.shift <- 200
df$logloss <- log(df$loss + output.shift)

# pick columns that correlate with output:
cor.cols <- list()
for (i in 1:ncol(df)) {
  cor.cols[[i]] <- cor.test(df[trainrows,"logloss"], as.numeric(df[trainrows,i]))$p.value
}
p.cols <- which(unlist(cor.cols) < 0.05)

# remove id and output:
cols.numeric <- setdiff(1:ncol(df), which(colnames(df) %in% c("id","loss", "logloss")))

# make column list:
cols <- colnames(df[,intersect(p.cols, cols.numeric)])

# add quadratic values for highly correlated columns
zero.cols <- which(unlist(cor.cols) == 0)
quad.cols <- c()
# add quadratic columns to df
for (col in zero.cols) {
  if (startsWith(colnames(df)[col],"cont")) {
    quad.col <- paste0("q", col)
    df[, quad.col] <- df[,col]^2
    test.df[, quad.col] <- test.df[,col]^2  
    quad.cols <- c(quad.cols, quad.col)
  }
}

cols <- c(cols, quad.cols)

df$multycat0 <- rep(0, nrow(df))
test.df$multycat0 <- rep(0, nrow(test.df))
for (col in zero.cols) {
  if (startsWith(colnames(df)[col],"cat")) {
    df$multycat0 <- df$multycat0 + as.numeric(df[,col])
    test.df$multycat0 <- test.df$multycat0 + as.numeric(test.df[,col])
  }
}

m.cols <- c("multycat0")
for (col in p.cols) {
  if (startsWith(colnames(df)[col],"cont")) {
    m.cols <- c(m.cols, colnames(df)[col])
  }
}


X_train <- df[trainrows, cols]
X_val <- df[valrows, cols]
Y_train <- df$logloss[trainrows]
Y_val <- df$logloss[valrows]
X_full <- df[,cols]
Y_full <- df$logloss

########################## FEATURE EXPLORATION ####################
library(caret)
# too small variance
zero.var <- nearZeroVar(X_train, saveMetrics=TRUE)

# Achtung! need to convert factors to numeric to plot features
num.df <- data.frame(data)
num.test.df <- data.frame(test.data)
for (i in 1:ncol(num.df)) {
  if (startsWith(colnames(num.df)[i], 'cat')) {
    num.df[,i] <- as.numeric(num.df[,i])
    num.test.df[,i] <- as.numeric(num.test.df[,i])
  }
}

featurePlot(num.df[trainrows,], num.df$loss[trainrows], "strip")
corrplot.mixed(cor(num.df[trainrows,cols]), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")

########################## XGBOOST #################################
library(xgboost)

xgb.full <- data.matrix(X_full)
xgb.full.label <- data.matrix(Y_full)

xgb.params <- list(max.depth = 11, 
                   eta = 0.01, 
                   min_child_weight = 1,
                   colsample_bytree = 1,
                   subsample = 0.9,
                   alpha = 1,
                   gamma = 2,
                   objective = "reg:linear",
                   eval_metric = maeeval,
                   base_score=7.76)

xgb.fit <- xgboost(data = xgb.full, label = xgb.full.label, params = xgb.params, nrounds = as.integer(2800/0.8), verbose = 1, print.every.n = 50, nthread = 2)
xgb.val.pred <- predict(xgb.fit, xgb.val)
xgb.val.mse <- sum((xgb.val.pred - Y_val)^2)/length(Y_val)


############# XGBOOST TRAIN ########################################
library(xgboost)
seed <- 309
maeeval <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- mean(abs(as.numeric(labels)- as.numeric(preds)))
  return(list(metric = "mae", value = err))
}

xgb.train <- data.matrix(X_train)
xgb.train.label <- data.matrix(Y_train)
xgb.val <- data.matrix(X_val)
xgb.val.label <- data.matrix(Y_val)
dtrain <- xgb.DMatrix(data = xgb.train, label = xgb.train.label)
dtest <- xgb.DMatrix(data = xgb.val, label = xgb.val.label)
watchlist <- list(test=dtest, train=dtrain)

xgb.full <- data.matrix(X_full)
xgb.full.label <- data.matrix(Y_full)
dfull <- xgb.DMatrix(data = xgb.full, label = xgb.full.label)

bestval_toString <- function(result,i) {
  return(paste0(Sys.time(), " Max depth:", result[[i]][1], 
               ", eta:", result[[i]][2],
               ", subsample:", result[[i]][3],
               ", child weight:", result[[i]][4],
               ", col sample by tree:", result[[i]][5],
               ", gamma:", result[[i]][6],
               ", rounds:", result[[i]][7],
               ", best val score (mae):", result[[i]][8]
  ))
}

best_cross_value <- c()
i <- 1
for (colsamplebytree in c(1, 0.9, 0.5)) {
  #print(paste(Sys.time(), "Start next colsample by tree:",colsamplebytree))
  for (subsample in c(0.9)) {
    #print(paste(Sys.time(), "Start next subsample:", subsample))
    for (childweight in c(1, 5)) {
      #print(paste(Sys.time(), "Start next min child weight:", childweight))
        for (eta in c(1)) {
          #print (paste(Sys.time(), "Start next eta:", eta))
          for (gamma in c(1, 0.8, 0.5, 0.2, 0)) {
            for (maxdepth in c(11)) {
            #print (paste(Sys.time(), "Start next maxdepth:", maxdepth))
              
              xgb.params <- list(max.depth = maxdepth, eta = eta, min_child_weight = childweight, colsample_bytree = colsamplebytree, subsample = subsample, alpha = 1, gamma = gamma, objective = "reg:linear", eval_metric = maeeval, seed=seed)
              
              nrounds <- 5001
              bst <- xgb.cv(data=dfull, params = xgb.params, nthread = 4, nround=nrounds, nfold=5, early.stop.round=7, maximize=FALSE, verbose = 1, print.every.n=as.integer(nrounds/100))
              
              nrounds <- which(bst$test.mae.mean == min(bst$test.mae.mean))
              bestscore <- paste0(min(bst$test.mae.mean),"+",bst$test.mae.std[nrounds])
              
              best_cross_value[[i]] <- c(maxdepth, eta, subsample, childweight, colsamplebytree, gamma, nrounds, bestscore)
              print(paste0(Sys.time(), bestval_toString(best_cross_value,i)))
              write(paste0(Sys.time(), bestval_toString(best_cross_value,i)), file="xgboost_params_test2.txt", append=TRUE)
              
              i<- i+1
            }
        }
      }
    }
  }
}

print(paste(Sys.time(), "Test completed. Scores are:"))
for (j in 1:length(best_cross_value)) {
  print(bestval_toString(best_cross_value,j))
}

# 1. max.depth vs child_weight: 11-5
# 2. gamma: 1
# 3. colsample_bytree and subsample:1-0.9
# 4. alpha: 1

# gamma = 1
# max.depth check 3:5 for large eta
# subsample = 0.9
# colsample_bytree
# child_weight



dfull <- xgb.DMatrix(data = xgb.full, label = xgb.full.label)
xgb.params <- list(max.depth = 11, 
                   eta = 0.1, 
                   min_child_weight = 5,
                   colsample_bytree = 0.5,
                   subsample = 0.9,
                   gamma = 0,
                   alpha = 1,
                   objective = "reg:linear",
                   eval_metric = maeeval,
                   seed=seed)

bst <- xgb.cv(data=dfull, params = xgb.params, nthread = 4, nround=3000, 
                 nfold=5, verbose = 1, print.every.n = 30,
                 early.stop.round=32, maximize=FALSE)

# max.depth = 4, eta = 1, nround = 20: mse=0.3235624
# max.depth = 3, eta = 0.1, nround = 1000: mse=0.2927492
# max.depth = 8, eta = 0.1, nround = 1000: mse=0.2961916
# max.depth = 12, eta = 0.01, nround = 1800: mse=0.2253347
# max.depth = 12, eta = 0.01, nround = 3000: mse=0.225966/ (0.2186261 on cross09.csv)
# 2424, 2277 - test mae 0.3708
# 2967 - test mae 0.2186261

# with files 08:
# max.depth = 8, eta=0.1, nround=128:    test-rmse=1851
# max.depth = 2, eta=1, nround=361:    test-rmse=1909
# nround=182, max.depth=12, eta=0.05: test-rmse=1866
# max.depth=3, eta=0.1, nround=500: test-rmse=1868
# max.depth=8, eta=0.03, nround=500: test_rmse=1841
# max.depth=3, eta=0.1, nround=5000: test_rmse=1642
# validation error maxdepth = 4, nround=20, eta=1 : 3896206
# validation error maxdepth = 2, nround=361: eta=1: 3645051
# validation error maxdepth = 3, nround=5000, eta=0.1: 2696021

############################ RUN TEST #####################################

xgb.test <- data.matrix(test.df[, cols])
xgb.test.predict <- predict(xgb.fit, xgb.test)
test.pred <- exp(xgb.test.predict) - output.shift
test.pred[which(test.pred < 0 )] <- 0
solution <- cbind(id=test.df$id, test.pred)
colnames(solution)[2] <- 'loss'

# check new solution has sense
solution.1132 <- read.csv('glm_claims_solution1132.csv')
sum(abs(solution.1132[,2] - solution[,2]))/sum(solution.1132[,2])
# must be pretty colse

write.csv(solution, file = 'glm_claims_solution.csv', row.names = F)

sol <- read.csv('glm_claims_solution.csv')
solution.1120 <- read.csv('glm_claims_solution1120.csv')
sum(abs(solution.1120[,2] - sol[,2]))/sum(solution.1120[,2])




################################### GLM #####################################
library(glmnet)
library(doParallel)
registerDoParallel(cores = 4)

# cols[which(!cols %in% features_to_drop)]
glm.train <- data.matrix(df[trainrows, cols])
glm.train.output <- data.matrix(df$logloss[trainrows])
glm.val <- data.matrix(df[valrows, cols])
glm.val.output <- data.matrix((df$logloss[valrows]))

# alpha 0-1 no difference in result
glm.cv <- cv.glmnet(glm.train, glm.train.output, alpha = 1, type.measure="mae", parallel = TRUE)

glm.val.predict <- predict(glm.cv, newx = glm.val, s="lambda.min")
glm.val.mae <- mean(abs(glm.val.predict - glm.val.output))

plot(glm.val.output, glm.val.predict, main="Validation set")

