library(pROC)
library(ggplot2)
library(keras)
library(dplyr)
library(caret)
library(tensorflow)
library(MASS)
library(class)
library(tree)
library(randomForest)
library(xgboost)
library(glmnet)
library(e1071)
library(ComplexHeatmap)
library(kernlab)
library(gbm)


###################### 
####################
#################
##############
############
#########
#######
###
##
all_result_acc <- list()
all_result_recall <- list()
all_result_FS <- list()
all_result_summary <- list()
all_result_npv <- list()
all_result_ppv <- list()
all_result_auc <- list() 
all_result_pre <- list() 
all_result_importance <- list()

####### 1.1 Lasso regression lambda.min
####
##
#
###
###
fold <- 10
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
#######
set.seed(123)
cvfit <- cv.glmnet(train_expd, labels,family = "binomial", nlambda=100, alpha=1,nfolds = fold)
cvfit$lambda.min
fit <- glmnet(train_expd,labels,family = "binomial")
coef <- coef(fit, s = cvfit$lambda.min)
lassomin<- row.names(coef)[which(coef != 0)]
lassomin
im <- as.data.frame(as.matrix(coef(fit, s = cvfit$lambda.min))[-1,])
colnames(im) <- "coefficients"
im[,1] <- abs(im)
im <- im[which(im$coefficients!=0),,drop=F]
all_result_importance[[paste0("LASSO labmda.min")]] <- im
pred.rr <- predict(cvfit,as.matrix(train_expd), s=cvfit$lambda.min,type="response")
roc_curve <- pROC::roc(train_labels, pred.rr)
roc_curve
auc_value <- auc(roc_curve)
levels_to_use <- c("negative", "positive")
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(cvfit, as.matrix(train_expd), s = cvfit$lambda.min, type = "response"))
colnames(train_result) <- "prediction"
train_result$type <- factor(ifelse(train_result$prediction > cuti, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, cvfit, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(cvfit, as.matrix(expdata), s = cvfit$lambda.min, type = "response"))
  colnames(tresult) <- "prediction"
  tresult$type <- factor(ifelse(tresult$prediction > cuti, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  return(tresult)
}, cvfit = cvfit, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("LASSO lambda.min")
result_metrics <- sapply(all_result, function(x) {
  x$predict_result <- factor(x$predict_result, levels = levels_to_use)
  x$real_label <- factor(x$real_label, levels = levels_to_use)
  conf_matrix <- confusionMatrix(x$real_label,x$predict_result, positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
result_metrics
# Bootstrap 
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(boot_train_expd, boot_labels,family = "binomial", nlambda=100, alpha=1,nfolds = fold)
  
  # 预测并计算指标
  pred_boot <- predict(cvfit, as.matrix(boot_test_expd), s = "lambda.min", type = "response")
  roc_curve <- pROC::roc(boot_test_labels , pred_boot)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
  real_labels_boot <- factor(ifelse(boot_test_labels  == 1, "positive", "negative"))
  pred_labels_boot <- factor(pred_labels_boot, levels = levels_to_use)
  # 计算混淆矩阵
  conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
  # 提取混淆矩阵的各项指标
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  # 计算 AUC
  roc_curve <- roc(real_labels_boot, as.numeric(pred_boot))
  auc_value <- auc(roc_curve)
  # 返回 Bootstrap 结果
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
# 计算 Bootstrap 平均结果
# 计算 Bootstrap 平均结果
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
# 将 Bootstrap 结果插入到 result_metrics 的第 3 列
result_metrics <- cbind(bootstrap_avg_metrics,result_metrics)
# MCCV 评估
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
set.seed(123)
n_mccv <- 100  # 设置 MCCV 评估次数
train_ratio <- 0.7  # 设置训练集比例
mccv_results <- lapply(1:n_mccv, function(i) {
  # MCCV 随机划分训练集和测试集
  #sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- labels[sample_idx]
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(mccv_train_expd, mccv_train_labels,family = "binomial", nlambda=100, alpha=1,nfolds = fold)
  
  # 预测并计算指标
  pred_mccv <- predict(cvfit, as.matrix(mccv_test_expd), s = "lambda.min", type = "response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
  real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
  pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
  # 计算混淆矩阵
  conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
  # 提取混淆矩阵的各项指标
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  # 计算 AUC
  roc_curve <- roc(real_labels_mccv, as.numeric(pred_mccv))
  auc_value <- auc(roc_curve)
  # 返回 MCCV 结果，包括 Brier Score
  metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  return(metrics)
})
# 将 MCCV 结果输出为列表
mccv_results_summary <- do.call(rbind, mccv_results)
mccv_results_summary <- as.data.frame(mccv_results_summary)
# 查看 MCCV 的结果统计
summary(mccv_results_summary)
averagemccv_metrics <- colMeans(mccv_results_summary, na.rm = TRUE)
result_metrics <- cbind(averagemccv_metrics,result_metrics)
# ###10折CV验证
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
# 设置模型的训练
set.seed(123)
model <- caret::train(
  labels ~ ., data = train_expd, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = cvfit$lambda.min),  
  family = "binomial",
  metric = "ROC",  # 使用 AUC 作为评估指标
  trControl = train_control
)
importance <- varImp(model)
#all_result_importance[[paste0("LASSO-CV lambda.min")]] <- as.data.frame(importance$importance)
resample_results <- model$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
# 提取模型的预测结果
predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
roc_curve <- roc(train_labels, predictions)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = c("negative", "positive"))
# 计算性能指标
conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy <- conf_matrix$overall['Accuracy']
recall <- conf_matrix$byClass['Recall']
precision <- conf_matrix$byClass['Precision']
f1_score <- conf_matrix$byClass['F1']
npv <- conf_matrix$byClass['Neg Pred Value']
ppv <- conf_matrix$byClass['Pos Pred Value']
# 返回结果
CV_metrics <- c(
  accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
  npv = npv, ppv = ppv, auc = auc_value
)
# 输出结果
print(CV_metrics)
result_metrics <- cbind(CV_metrics,result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
# 提取结果并存储到对应列表中
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
all_result_summary[[result_label]] <- all_result
result_metrics
####### 1.2  Lasso regression  lambda.1se
####
##
#

train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
#######
set.seed(123)
cvfit <- cv.glmnet(train_expd, labels,family = "binomial", nlambda=100, alpha=1,nfolds = fold)
cvfit$lambda.1se
fit <- glmnet(train_expd,labels,family = "binomial")
coef <- coef(fit, s = cvfit$lambda.1se)
lasso1se<- row.names(coef)[which(coef != 0)]
lasso1se
im <- as.data.frame(as.matrix(coef(fit, s = cvfit$lambda.1se))[-1,])
colnames(im) <- "coefficients"
im[,1] <- abs(im)
im <- im[which(im$coefficients!=0),,drop=F]
all_result_importance[[paste0("LASSO lambda.1se")]] <- im
pred.rr <- predict(cvfit,as.matrix(train_expd), s=cvfit$lambda.1se,type="response")
roc_curve <- pROC::roc(train_labels, pred.rr)
roc_curve
auc_value <- auc(roc_curve)
levels_to_use <- c("negative", "positive")
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(cvfit, as.matrix(train_expd), s = cvfit$lambda.1se, type = "response"))
colnames(train_result) <- "prediction"
train_result$type <- factor(ifelse(train_result$prediction > cuti, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, cvfit, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(cvfit, as.matrix(expdata), s = cvfit$lambda.1se, type = "response"))
  colnames(tresult) <- "prediction"
  tresult$type <- factor(ifelse(tresult$prediction > cuti, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  return(tresult)
}, cvfit = cvfit, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("LASSO lambda.1se")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_label,x$predict_result, positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
result_metrics
# Bootstrap 
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(boot_train_expd , boot_labels,family = "binomial", nlambda=100, alpha=1,nfolds = fold)
  pred_boot <- predict(cvfit, as.matrix(boot_test_expd), s = "lambda.1se", type = "response")
  roc_curve <- pROC::roc(boot_test_labels, pred_boot)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
  real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
  pred_labels_boot <- factor(pred_labels_boot, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_boot, as.numeric(pred_boot))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics,result_metrics)
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- labels[sample_idx]
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(mccv_train_expd, mccv_train_labels,family = "binomial", nlambda=100, alpha=1,nfolds = fold)
  pred_mccv <- predict(cvfit, as.matrix(mccv_test_expd), s = "lambda.1se", type = "response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
  real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
  pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_mccv, as.numeric(pred_mccv))
  auc_value <- auc(roc_curve)
  metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  return(metrics)
})
mccv_results_summary <- do.call(rbind, mccv_results)
mccv_results_summary <- as.data.frame(mccv_results_summary)
summary(mccv_results_summary)
averagemccv_metrics <- colMeans(mccv_results_summary, na.rm = TRUE)
result_metrics <- cbind(averagemccv_metrics,result_metrics)
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
set.seed(123)
model <- caret::train(
  labels ~ ., data = train_expd, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = cvfit$lambda.1se), 
  family = "binomial",
  metric = "ROC", 
  trControl = train_control
)
importance <- varImp(model)
resample_results <- model$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
roc_curve <- roc(train_labels, predictions)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = c("negative", "positive"))
conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy <- conf_matrix$overall['Accuracy']
recall <- conf_matrix$byClass['Recall']
precision <- conf_matrix$byClass['Precision']
f1_score <- conf_matrix$byClass['F1']
npv <- conf_matrix$byClass['Neg Pred Value']
ppv <- conf_matrix$byClass['Pos Pred Value']
CV_metrics <- c(
  accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
  npv = npv, ppv = ppv, auc = auc_value
)
print(CV_metrics)
result_metrics <- cbind(CV_metrics,result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
all_result_summary[[result_label]] <- all_result
result_metrics
####### 2.1  ridge regression & lambda.min
####
##
#

train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
#######
set.seed(123)
cvfit <- cv.glmnet(train_expd, labels,family = "binomial", nlambda=100, alpha=0,nfolds = fold)
best_lambda <- cvfit$lambda.min  
coef_matrix <- coef(cvfit, s = best_lambda)
coef_df <- as.data.frame(as.matrix(coef_matrix))
colnames(coef_df) <- "coefficients"
coef_df <- coef_df[rownames(coef_df) != "(Intercept)", , drop=FALSE]
non_zero_coef_df <- coef_df[coef_df$coefficients != 0, , drop=FALSE]
print(non_zero_coef_df)
all_result_importance[[paste0("RR lambda.min")]] <- non_zero_coef_df
pred.rr <- predict(cvfit,as.matrix(train_expd), s=cvfit$lambda.min,type="response")
roc_curve <- pROC::roc(train_labels, pred.rr)
roc_curve
auc_value <- auc(roc_curve)
levels_to_use <- c("negative", "positive")
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(cvfit, as.matrix(train_expd), s = cvfit$lambda.min, type = "response"))
colnames(train_result) <- "prediction"
train_result$type <- factor(ifelse(train_result$prediction > cuti, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, cvfit, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(cvfit, as.matrix(expdata), s = cvfit$lambda.min, type = "response"))
  colnames(tresult) <- "prediction"
  tresult$type <- factor(ifelse(tresult$prediction > cuti, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  return(tresult)
}, cvfit = cvfit, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("RR lambda.min")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
# Bootstrap 
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(boot_train_expd, boot_labels,family = "binomial", nlambda=100, alpha=0,nfolds = fold)
  pred_boot <- predict(cvfit, as.matrix(boot_test_expd), s = "lambda.min", type = "response")
  roc_curve <- pROC::roc(boot_test_labels, pred_boot)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
  real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
  pred_labels_boot <- factor(pred_labels_boot, levels = levels_to_use)
  conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_boot, as.numeric(pred_boot))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics,result_metrics)
set.seed(123)
n_mccv <- 100  #
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- labels[sample_idx]
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(mccv_train_expd, mccv_train_labels,family = "binomial", nlambda=100, alpha=0,nfolds = fold)
  pred_mccv <- predict(cvfit, as.matrix(mccv_test_expd), s = "lambda.min", type = "response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
  real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
  pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix( real_labels_mccv,pred_labels_mccv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_mccv, as.numeric(pred_mccv))
  auc_value <- auc(roc_curve)
  metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  return(metrics)
})
mccv_results_summary <- do.call(rbind, mccv_results)
mccv_results_summary <- as.data.frame(mccv_results_summary)
summary(mccv_results_summary)
averagemccv_metrics <- colMeans(mccv_results_summary, na.rm = TRUE)
result_metrics <- cbind(averagemccv_metrics,result_metrics)

train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
set.seed(123)
model <- caret::train(
  labels ~ ., data = train_expd, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = cvfit$lambda.min),
  family = "binomial",
  metric = "ROC",  
  trControl = train_control
)
importance <- varImp(model)
resample_results <- model$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
roc_curve <- roc(train_labels, predictions)

optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = c("negative", "positive"))
conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy <- conf_matrix$overall['Accuracy']
recall <- conf_matrix$byClass['Recall']
precision <- conf_matrix$byClass['Precision']
f1_score <- conf_matrix$byClass['F1']
npv <- conf_matrix$byClass['Neg Pred Value']
ppv <- conf_matrix$byClass['Pos Pred Value']
CV_metrics <- c(
  accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
  npv = npv, ppv = ppv, auc = auc_value
)
print(CV_metrics)
result_metrics <- cbind(CV_metrics,result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
all_result_summary[[result_label]] <- all_result
result_metrics

####### 2.2  ridge regression & lambda.1se
####
##
#

train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
#######
set.seed(123)
cvfit <- cv.glmnet(train_expd, labels,family = "binomial", nlambda=100, alpha=0,nfolds = fold)
cvfit$lambda.1se
fit <- glmnet(train_expd,labels,family = "binomial")
best_lambda <- cvfit$lambda.1se  
coef_matrix <- coef(cvfit, s = best_lambda)
coef_df <- as.data.frame(as.matrix(coef_matrix))
colnames(coef_df) <- "coefficients"
coef_df <- coef_df[rownames(coef_df) != "(Intercept)", , drop=FALSE]
non_zero_coef_df <- coef_df[coef_df$coefficients != 0, , drop=FALSE]
print(non_zero_coef_df)
all_result_importance[[paste0("RR lambda.1se")]] <- non_zero_coef_df
pred.rr <- predict(cvfit,as.matrix(train_expd), s=cvfit$lambda.1se,type="response")
roc_curve <- pROC::roc(train_labels, pred.rr)
roc_curve
auc_value <- auc(roc_curve)
levels_to_use <- c("negative", "positive")
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(cvfit, as.matrix(train_expd), s = cvfit$lambda.1se, type = "response"))
colnames(train_result) <- "prediction"
train_result$type <- factor(ifelse(train_result$prediction > cuti, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, cvfit, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(cvfit, as.matrix(expdata), s = cvfit$lambda.1se, type = "response"))
  colnames(tresult) <- "prediction"
  tresult$type <- factor(ifelse(tresult$prediction > cuti, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  return(tresult)
}, cvfit = cvfit, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result

result_label <- paste0("RR lambda.1se")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
# Bootstrap 
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(boot_train_expd, boot_labels,family = "binomial", nlambda=100, alpha=0,nfolds = fold)
  pred_boot <- predict(cvfit, as.matrix(boot_test_expd), s = "lambda.1se", type = "response")
  roc_curve <- pROC::roc(boot_test_labels, pred_boot)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
  real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
  pred_labels_boot <- factor(pred_labels_boot, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_boot, as.numeric(pred_boot))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})

bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics,result_metrics)
# MCCV 
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- labels[sample_idx]
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(mccv_train_expd, mccv_train_labels,family = "binomial", nlambda=100, alpha=0,nfolds = fold)
  pred_mccv <- predict(cvfit, as.matrix(mccv_test_expd), s = "lambda.1se", type = "response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
  real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
  pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_mccv, as.numeric(pred_mccv))
  auc_value <- auc(roc_curve)
  metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  return(metrics)
})
mccv_results_summary <- do.call(rbind, mccv_results)
mccv_results_summary <- as.data.frame(mccv_results_summary)
summary(mccv_results_summary)
averagemccv_metrics <- colMeans(mccv_results_summary, na.rm = TRUE)
result_metrics <- cbind(averagemccv_metrics,result_metrics)


train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
set.seed(123)
model <- caret::train(
  labels ~ ., data = train_expd, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = cvfit$lambda.1se),  
  family = "binomial",
  metric = "ROC",  
  trControl = train_control
)
importance <- varImp(model)
resample_results <- model$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
roc_curve <- roc(train_labels, predictions)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy <- conf_matrix$overall['Accuracy']
recall <- conf_matrix$byClass['Recall']
precision <- conf_matrix$byClass['Precision']
f1_score <- conf_matrix$byClass['F1']
npv <- conf_matrix$byClass['Neg Pred Value']
ppv <- conf_matrix$byClass['Pos Pred Value']
CV_metrics <- c(
  accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
  npv = npv, ppv = ppv, auc = auc_value
)
print(CV_metrics)
result_metrics <- cbind(CV_metrics,result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics
####### 3.1. Elastic Net regression  lambda.min
####
##
#
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
alpha_values <- seq(0.1, 0.9, by = 0.1)
cv_results <- list()
for (alphai in alpha_values) {
  set.seed(123)  
  cvfit <- cv.glmnet(train_expd, labels, family = "binomial", alpha = alphai, nlambda = 100, nfolds = fold)
  cv_results[[as.character(alphai)]] <- cvfit$cvm[which(cvfit$lambda == cvfit$lambda.min)] 
}
optimal_alpha <- alpha_values[which.min(unlist(cv_results))]
#######
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
set.seed(123)
cvfit <- cv.glmnet(train_expd, labels,family = "binomial", nlambda=100, alpha=optimal_alpha,nfolds = fold)
cvfit$lambda.min
fit <- glmnet(train_expd,labels,family = "binomial")
coef <- coef(fit, s = cvfit$lambda.min)
ENRmin <- row.names(coef)[which(coef != 0)]
ENRmin
im <- as.data.frame(as.matrix(coef(fit, s = cvfit$lambda.min))[-1,])
colnames(im) <- "coefficients"
im[,1] <- abs(im)
im <- im[which(im$coefficients!=0),,drop=F]
all_result_importance[[paste0("ENR lambda.min")]] <- im
pred.rr <- predict(cvfit,as.matrix(train_expd), s=cvfit$lambda.min,type="response")
roc_curve <- pROC::roc(train_labels, pred.rr)
roc_curve
auc_value <- auc(roc_curve)
levels_to_use <- c("negative", "positive")
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(cvfit, as.matrix(train_expd), s = cvfit$lambda.min, type = "response"))
colnames(train_result) <- "prediction"
train_result$type <- factor(ifelse(train_result$prediction > cuti, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, cvfit, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(cvfit, as.matrix(expdata), s = cvfit$lambda.min, type = "response"))
  colnames(tresult) <- "prediction"
  tresult$type <- factor(ifelse(tresult$prediction > cuti, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  return(tresult)
}, cvfit = cvfit, labelsdata = all_labels)

all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("ENR lambda.min")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
# Bootstrap 
set.seed(123)
n_bootstrap <- 1000 
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(boot_train_expd, boot_labels,family = "binomial", nlambda=100, alpha=optimal_alpha,nfolds = fold)
  pred_boot <- predict(cvfit, as.matrix(boot_test_expd), s = "lambda.min", type = "response")
  roc_curve <- pROC::roc(boot_test_labels, pred_boot)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
  real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
  pred_labels_boot <- factor(pred_labels_boot, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_boot, as.numeric(pred_boot))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics,result_metrics)
# MCCV 
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- labels[sample_idx]
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(mccv_train_expd, mccv_train_labels,family = "binomial", nlambda=100, alpha=optimal_alpha,nfolds = fold)
  pred_mccv <- predict(cvfit, as.matrix(mccv_test_expd), s = "lambda.min", type = "response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
  real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
  pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_mccv, as.numeric(pred_mccv))
  auc_value <- auc(roc_curve)
  metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  return(metrics)
})
mccv_results_summary <- do.call(rbind, mccv_results)
mccv_results_summary <- as.data.frame(mccv_results_summary)
summary(mccv_results_summary)
averagemccv_metrics <- colMeans(mccv_results_summary, na.rm = TRUE)
result_metrics <- cbind(averagemccv_metrics,result_metrics)

train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
set.seed(123)
model <- caret::train(
  labels ~ ., data = train_expd, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = optimal_alpha, lambda = cvfit$lambda.min), 
  family = "binomial",
  metric = "ROC", 
  trControl = train_control
)
importance <- varImp(model)
resample_results <- model$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
roc_curve <- roc(train_labels, predictions)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = c("negative", "positive"))
conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy <- conf_matrix$overall['Accuracy']
recall <- conf_matrix$byClass['Recall']
precision <- conf_matrix$byClass['Precision']
f1_score <- conf_matrix$byClass['F1']
npv <- conf_matrix$byClass['Neg Pred Value']
ppv <- conf_matrix$byClass['Pos Pred Value']
CV_metrics <- c(
  accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
  npv = npv, ppv = ppv, auc = auc_value
)
print(CV_metrics)
result_metrics <- cbind(CV_metrics,result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
all_result_summary[[result_label]] <- all_result
result_metrics



####### 3.2. Elastic Net regression  lambda.1se
####
##
#
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
alpha_values <- seq(0.1, 0.9, by = 0.1)
cv_results <- list()
for (alphai in alpha_values) {
  set.seed(1)  
  cvfit <- cv.glmnet(train_expd, labels, family = "binomial", alpha = alphai, nlambda = 100, nfolds = fold)
  cv_results[[as.character(alphai)]] <- cvfit$cvm[which(cvfit$lambda == cvfit$lambda.1se)] 
}
optimal_alpha <- alpha_values[which.min(unlist(cv_results))]
#######optimal_alpha <- 0.4
train_expd <- as.matrix(train_exp)
labels <- factor(train_labels)
set.seed(123)
cvfit <- cv.glmnet(train_expd, labels,family = "binomial", nlambda=100, alpha=optimal_alpha,nfolds = fold)
cvfit$lambda.1se
fit <- glmnet(train_expd,labels,family = "binomial")
coef <- coef(fit, s = cvfit$lambda.1se)
ENR1se <- row.names(coef)[which(coef != 0)]
ENR1se
im <- as.data.frame(as.matrix(coef(fit, s = cvfit$lambda.1se))[-1,])
colnames(im) <- "coefficients"
im[,1] <- abs(im)
im <- im[which(im$coefficients!=0),,drop=F]
all_result_importance[[paste0("ENR lambda.1se")]] <- im
pred.rr <- predict(cvfit,as.matrix(train_expd), s=cvfit$lambda.1se,type="response")
roc_curve <- pROC::roc(train_labels, pred.rr)
roc_curve
auc_value <- auc(roc_curve)
levels_to_use <- c("negative", "positive")
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(cvfit, as.matrix(train_expd), s = cvfit$lambda.1se, type = "response"))
colnames(train_result) <- "prediction"
train_result$type <- factor(ifelse(train_result$prediction > cuti, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, cvfit, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(cvfit, as.matrix(expdata), s = cvfit$lambda.1se, type = "response"))
  colnames(tresult) <- "prediction"
  tresult$type <- factor(ifelse(tresult$prediction > cuti, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  return(tresult)
}, cvfit = cvfit, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("ENR lambda.1se")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
# Bootstrap 
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(boot_train_expd, boot_labels,family = "binomial", nlambda=100, alpha=optimal_alpha,nfolds = fold)
  pred_boot <- predict(cvfit, as.matrix(boot_test_expd), s = "lambda.1se", type = "response")
  roc_curve <- pROC::roc(boot_test_labels, pred_boot)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
  real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
  pred_labels_boot <- factor(pred_labels_boot, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_boot, as.numeric(pred_boot))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics,result_metrics)
# MCCV 
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7 
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- labels[sample_idx]
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- labels[-sample_idx]
  cvfit <- cv.glmnet(mccv_train_expd,mccv_train_labels,family = "binomial", nlambda=100, alpha=optimal_alpha,nfolds = fold)
  pred_mccv <- predict(cvfit, as.matrix(mccv_test_expd), s = "lambda.1se", type = "response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
  real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
  pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(real_labels_mccv, as.numeric(pred_mccv))
  auc_value <- auc(roc_curve)
  metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  return(metrics)
})
mccv_results_summary <- do.call(rbind, mccv_results)
mccv_results_summary <- as.data.frame(mccv_results_summary)
summary(mccv_results_summary)
averagemccv_metrics <- colMeans(mccv_results_summary, na.rm = TRUE)
result_metrics <- cbind(averagemccv_metrics,result_metrics)

train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)

set.seed(123)
model <- caret::train(
  labels ~ ., data = train_expd, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = optimal_alpha, lambda = cvfit$lambda.1se),  
  family = "binomial",
  metric = "ROC",  
  trControl = train_control
)
resample_results <- model$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
importance <- varImp(model)
predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
roc_curve <- roc(train_labels, predictions)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy <- conf_matrix$overall['Accuracy']
recall <- conf_matrix$byClass['Recall']
precision <- conf_matrix$byClass['Precision']
f1_score <- conf_matrix$byClass['F1']
npv <- conf_matrix$byClass['Neg Pred Value']
ppv <- conf_matrix$byClass['Pos Pred Value']
CV_metrics <- c(
  accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
  npv = npv, ppv = ppv, auc = auc_value
)
print(CV_metrics)
result_metrics <- cbind(CV_metrics,result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics

##############################################################################
#######################################' 4. Logistic regression
##############################################################################
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
model <- glm(labels ~ ., family = "binomial",data = train_expd)
importancei <- as.data.frame(model$coefficients[-1])
colnames(importancei) <- "coefficients"
all_result_importance[[paste0("LR")]] <- importancei
train_probs <- predict(model, type = "response")
roc_curve <- roc(train_labels, train_probs)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
levels_to_use <- c("positive", "negative")
train_probs <- predict(model, type = "response")
roc_curve <- roc(train_labels, train_probs)
auc_value <- auc(roc_curve)
train_result <- as.data.frame(predict(model, type="response"))
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, expdata, type = "response"))
  tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = c("negative","positive"))
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = c("negative","positive"))
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result

result_label <- paste0("LR")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
# Bootstrap 
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_labels[-sample_idx]
  model <- glm(labels ~ ., family = "binomial", data = boot_train_expd)
  pred_boot <- predict(model, newdata = boot_test_expd, type = "response")
  roc_curve <- pROC::roc(boot_test_labels, pred_boot)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
  real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
  pred_labels_boot <- factor(pred_labels_boot, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
# MCCV
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- train_labels[sample_idx]
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- train_labels[-sample_idx]
  model <- glm(labels ~ ., family = "binomial", data = mccv_train_expd)
  pred_mccv <- predict(model, newdata = mccv_test_expd, type = "response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
  real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
  pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})

mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)

train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
set.seed(123)
model_cv <- caret::train(
  labels ~ ., data = train_expd, 
  method = "glm",
  family = "binomial",
  metric = "ROC",  
  trControl = train_control
)
resample_results <- model_cv$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
importance <- varImp(model_cv)
importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "Overall"])
importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
rownames(importance_filtered) <- importance_filtered$Feature
importance_filtered <- importance_filtered[ , -1, drop = FALSE]
predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
roc_curve_cv <- pROC::roc(train_labels, predictions_cv)
optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
cuti_cv <- optimal_coords_cv$threshold
pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")

accuracy_cv <- conf_matrix_cv$overall['Accuracy']
recall_cv <- conf_matrix_cv$byClass['Recall']
precision_cv <- conf_matrix_cv$byClass['Precision']
f1_score_cv <- conf_matrix_cv$byClass['F1']
npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
cv_metrics <- c(
  accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
  f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
all_result_summary[[result_label]] <- all_result
result_metrics

##############################################################################
#######################################' 5. Linear discriminant analysis
##############################################################################
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
set.seed(123)
model <- lda(labels ~ ., data = train_expd)
all_result_importance[[paste0("LDA")]] <- as.data.frame(model$scaling)
non_zero_features <- rownames(model$scaling)[model$scaling != 0]
LDA <- non_zero_features
train_predict <- predict(model, train_expd)
train_probs <- train_predict$posterior[,2]  
roc_curve <- roc(train_expd$labels, train_probs)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
roc_curve <- roc(train_labels, train_probs)
auc_value <- auc(roc_curve)
train_result <- as.data.frame(train_probs)
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, expdata)$posterior[, 2])
  tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("LDA")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p),levels = c("negative", "positive"), direction = "<"))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
#BOOSTRAP
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_labels[-sample_idx]
  model <- lda(labels ~ ., data = boot_train_expd)
  pred_probs <- predict(model, newdata = boot_test_expd)$posterior[, 2]
  roc_curve <- pROC::roc(boot_test_labels, pred_probs,levels = c(0, 1), direction = "<")
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  boot_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
  levels(boot_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(boot_labels,pred_labels, positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
set.seed(123)
n_mccv <- 100 
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- mccv_train_expd$labels
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- mccv_test_expd$labels
  model <- lda(labels ~ ., data = mccv_train_expd)
  pred_probs <- predict(model, newdata = mccv_test_expd)$posterior[, 2]
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs,levels = c(0, 1), direction = "<")
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
  levels(mccv_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- mccv_test_labels
  pred_labels <- factor(pred_labels, levels = levels_to_use)
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#cv
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
set.seed(123)
model_cv <- caret::train(
  labels ~ ., data = train_expd, 
  method = "lda",
  metric = "ROC",  
  trControl = train_control
)
resample_results <- model_cv$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
importance <- varImp(model_cv)
plot(importance)
importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
rownames(importance_filtered) <- importance_filtered$Feature
importance_filtered <- importance_filtered[ , -1, drop = FALSE]
predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv,levels = c("negative", "positive"), direction = "<")
optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
cuti_cv <- optimal_coords_cv$threshold
pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy_cv <- conf_matrix_cv$overall['Accuracy']
recall_cv <- conf_matrix_cv$byClass['Recall']
precision_cv <- conf_matrix_cv$byClass['Precision']
f1_score_cv <- conf_matrix_cv$byClass['F1']
npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
cv_metrics <- c(
  accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
  f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics

##############################################################################
#######################################' 6.k-Nearest Neighbor
##############################################################################
train_expd <- as.data.frame(train_exp)
train_expd<- train_expd
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
###
knumber <- 1:100 
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
set.seed(123)
model <- caret::train(
  labels ~ .,
  data = train_expd,
  method = "knn",
  trControl = train_control,
  tuneGrid = data.frame(k =knumber) 
)
print(model$bestTune)
model <- caret::train(
  labels ~ .,
  data = train_expd,
  method = "knn",
  trControl = train_control,
  tuneGrid = data.frame(k =70) 
)
importance <- varImp(model)
print(importance)
plot(importance)
importance <- varImp(model)
importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
rownames(importance_filtered) <- importance_filtered$Feature
importance_filtered <- importance_filtered[ , -1, drop = FALSE]
all_result_importance[[paste0("KNN")]] <- as.data.frame(importance_filtered)
KNN  <- rownames(importance_filtered)
best_k <- model$bestTune$k
train_probs <- predict(model, type="prob")[,2]
roc_curve <- roc(train_expd$labels, train_probs)
roc_curve
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti<- optimal_coords$threshold
#####
train_result <- as.data.frame(predict(model, type="prob")[,2])
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, expdata, type = "prob")[,2])
  tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("KNN")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
result_metrics
#BOOTSTRAP
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_expd$labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_expd$labels[-sample_idx]
  model <- caret::train(
    labels ~ .,
    data = boot_train_expd,
    method = "knn",
    trControl = train_control,
    tuneGrid = data.frame(k =70)
  )
  pred_probs <- predict(model, newdata =  boot_test_expd,type="prob")[,2]
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  levels(boot_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- boot_test_labels
  pred_labels <- factor(pred_labels, levels = levels_to_use)
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- mccv_train_expd$labels
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- mccv_test_expd$labels
  model <- caret::train(
    labels ~ .,
    data = mccv_train_expd,
    method = "knn",
    trControl = train_control,
    tuneGrid = data.frame(k = 70)
  )
  pred_probs <- predict(model, newdata = mccv_test_expd, type = "prob")[, 2]
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  mccv_test_labels <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"), levels = levels_to_use)
  pred_labels <- factor(ifelse(pred_probs > cuti, "positive", "negative"), levels = levels_to_use)
  conf_matrix <- confusionMatrix(mccv_test_labels,pred_labels, positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#cv
train_expd <- as.data.frame(train_exp)
train_expd<- train_expd
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
levels(train_expd$labels)
set.seed(123)
model <- caret::train(
  labels ~ .,
  data = train_expd,
  method = "knn",metric = "ROC",
  trControl = train_control,
  tuneGrid = data.frame(k =70)
)
importance <- varImp(model)
resample_results <- model$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
roc_curve <- roc(train_labels, predictions)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = c("negative", "positive"))
conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy <- conf_matrix$overall['Accuracy']
recall <- conf_matrix$byClass['Recall']
precision <- conf_matrix$byClass['Precision']
f1_score <- conf_matrix$byClass['F1']
npv <- conf_matrix$byClass['Neg Pred Value']
ppv <- conf_matrix$byClass['Pos Pred Value']
CV_metrics <- c(
  accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
  npv = npv, ppv = ppv, auc = auc_value
)
print(CV_metrics)
result_metrics <- cbind(CV_metrics,result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics

##############################################################################
#######################################' 7.Decision tree
##############################################################################
set.seed(123)
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
control_params <- tree.control(
  nobs = nrow(train_expd),   
  mincut = 7,                
  minsize = 40,              
  mindev = 0.01 )

model <- tree(labels ~ . - labels, data = train_expd, control = control_params)
summary(model)

used_features <- model$frame$var %>% 
  as.character() %>%        
  .[. != "<leaf>"] %>%      
  unique() 
importance_values <- numeric(length(used_features))
names(importance_values) <- used_features
non_leaf_nodes <- which(model$frame$var != "<leaf>")
for (i in non_leaf_nodes) {
  feature <- as.character(model$frame$var[i])

  dev_reduction <- model$frame$dev[i] - sum(model$frame$dev[model$children[[i]]])
  importance_values[feature] <- importance_values[feature] + dev_reduction
}
importance_df <- data.frame(
  Overall = importance_values,
  row.names = names(importance_values)
) %>% 
  arrange(-Overall) %>% 
  filter(Overall > 0)
all_result_importance[[paste0("DT")]] <- importance_df
print(importance_df)
DT <- rownames(importance_df)
DT
predictions <- predict(model, newdata = train_expd, type = "vector")[,2]
roc_DT <- pROC::roc(train_labels, predictions)
roc_DT
optimal_coords <- coords(roc_DT, "best", best.method = "youden", ret = "all")
cuti<- optimal_coords$threshold
train_result <- as.data.frame(predict(model, type="vector"))
train_result$type <- factor(ifelse(train_result[,2] > cuti, "positive", "negative"))
train_result <- train_result[,-1]
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, expdata, type = "vector")[,2])
  tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("DT")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)),levels = c("negative", "positive"),  # 指定类别顺序
                   direction = "<")
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
result_metrics
#BOOTSTRAP
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_expd$labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_expd$labels[-sample_idx]
  model <- tree(labels ~ . - labels, data = boot_train_expd, control = control_params)
  pred_probs <- predict(model, newdata = boot_test_expd,type="vector")[,2]
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels, positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#MCCV
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- mccv_train_expd$labels
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- mccv_test_expd$labels
  model <- tree(labels ~ . - labels, data = mccv_train_expd, control = control_params)
  pred_probs <- predict(model, newdata = mccv_test_expd,type="vector")[,2]
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- mccv_test_labels
  pred_labels <- factor(pred_labels, levels = levels_to_use)
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#cv
set.seed(123)
folds <- createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
cv_results <- list()
for (i in seq_along(folds)) {
  train_idx <- folds[[i]]
  train_data <- train_expd[train_idx, ]
  train_test_data <- train_expd[-train_idx, ]
  model <- tree(labels ~ . - labels, data = train_data, control = control_params)
  predictions_cv <- predict(model, newdata = train_test_data, type = "vector")[,2]
  roc_curve_cv <- pROC::roc(train_test_data$labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_test_data$labels == 1, "positive", "negative"))
  #real_labels_cv <- train_test_data$labels
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve_cv)
  cv_results[[i]] <- c(
    accuracy = as.numeric(accuracy_cv), 
    recall = as.numeric(recall_cv), 
    precision = as.numeric(precision_cv), 
    f1_score = as.numeric(f1_score_cv), 
    npv = as.numeric(npv_cv), 
    ppv = as.numeric(ppv_cv), 
    auc = as.numeric(auc_value)
  )
}
cv_results_df <- do.call(rbind, cv_results)
mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
cv_results <- list()
cv_metrics <- c(
  accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
  f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
all_result_summary[[result_label]] <- all_result
result_metrics

##############################################################################
#######################################' 8.Random forest
##############################################################################
test_expd <- as.matrix(test_exp)
test_labels <- factor(test_labels)
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
ctrl <- trainControl(method = "cv", number = 5,classProbs = TRUE,    
                     summaryFunction = twoClassSummary, 
                     verboseIter = TRUE)
grid <- expand.grid(
  mtry = seq(1, 50, 1)  
)

set.seed(123)
weights <- ifelse(train_expd$labels == "positive", 8, 1)
model <- caret::train(labels ~ .,
                      data = train_expd,
                      method = "rf",  
                      trControl = ctrl,
                      tuneGrid = grid,weights = weights)
print(model$bestTune)
grid <- expand.grid(mtry =7)
set.seed(123)
weights <- ifelse(train_expd$labels == "positive", 8, 1)
model <- caret::train(labels ~ .,
                      data = train_expd,
                      method = "rf",   
                      trControl = ctrl,
                      tuneGrid = grid,ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
pred.RF <- predict(model,newdata=train_expd,type="prob")
roc_RFtrain <- pROC::roc(train_expd$labels, pred.RF[, 2])
roc_RFtrain
pred.RFtest <- predict(model,newdata=test_expd,type="prob")
roc_RFtest <- pROC::roc(test_labels, pred.RFtest[, 2])
roc_RFtest
importance <- varImp(model)
plot(importance)
print(importance)
non_zero_importance <- as.data.frame(importance$importance)
non_zero_importance <- non_zero_importance[non_zero_importance$Overall > 0, , drop = FALSE]  # 筛选重要性大于0的特征
all_result_importance[[paste0("RF")]] <- non_zero_importance
RF <- rownames(non_zero_importance)
RF

train_probs <- predict(model, type="prob")[,2]
roc_curve <- roc(train_expd$labels, train_probs)
roc_curve
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti<- optimal_coords$threshold
#####
train_result <- as.data.frame(predict(model, type="prob")[,2])
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, expdata, type = "prob")[,2])
  tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result

result_label <- paste0("RF")
result_metrics <- sapply(all_result, function(x) {
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
#BT
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_expd$labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- boot_test_expd$labels
  weights <- ifelse(boot_train_expd$labels == "positive", 8, 1)
  model <- caret::train(labels ~ .,
                        data = boot_train_expd,
                        method = "rf",
                        trControl = ctrl,
                        tuneGrid = grid,
                        ntree= 300, sampsize = 350,nodesize = 30,
                        weights = weights)
  pred_probs <- predict(model, newdata = boot_test_expd, type = "prob")[, 2]
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- boot_test_labels
  pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
  real_labels <- factor(real_labels, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels, pred_labels, positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_results <- Filter(Negate(is.null), bootstrap_results)
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- mccv_train_expd$labels
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- mccv_test_expd$labels
  weights <- ifelse(mccv_train_expd$labels == "positive", 8, 1)
  model <- caret::train(labels ~ .,
                        data = mccv_train_expd,
                        method = "rf",   
                        trControl = ctrl,
                        tuneGrid = grid,ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
  pred_probs <- predict(model, newdata = mccv_test_expd,type="prob")[,2]
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- mccv_test_labels
  pred_labels <- factor(pred_labels, levels = levels_to_use)
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#cv
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
set.seed(123)
folds <- createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
cv_results <- list()
for (i in seq_along(folds)) {
  train_idx <- folds[[i]]
  train_data <- train_expd[train_idx, ]
  train_labels_CV <- train_labels[train_idx]
  test_data <- train_expd[-train_idx, ]
  CV_labels <- train_labels[-train_idx]
  weights <- ifelse(train_data$labels == "positive", 8, 1)
  model <- caret::train(labels ~ .,
                        data = train_data,
                        method = "rf",  
                        trControl = ctrl,
                        tuneGrid = grid,ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
  predictions_cv <- predict(model, newdata =  test_data, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(CV_labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(CV_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve_cv)
  cv_results[[i]] <- c(
    accuracy = as.numeric(accuracy_cv), 
    recall = as.numeric(recall_cv), 
    precision = as.numeric(precision_cv), 
    f1_score = as.numeric(f1_score_cv), 
    npv = as.numeric(npv_cv), 
    ppv = as.numeric(ppv_cv), 
    auc = as.numeric(auc_value)
  )
}
cv_results_df <- do.call(rbind, cv_results)
mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
cv_results <- list()
cv_metrics <- c(
  accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
  f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ] 
all_result_summary[[result_label]] <- all_result
result_metrics
##############################################################################
#######################################' 9.XGBOOST
##############################################################################
test_exp_names <- "DatasetB.txt"
test_exp <- exp_list[[which(exp_file==test_exp_names)]]
comtest <- intersect(rownames(test_exp), rownames(all_labels))
test_labels <- all_labels[comtest,]
test_exp <-test_exp[comtest,]
test_labels <- test_labels[, 2]
test_expd <- as.data.frame(test_exp)
test_expd$labels <- test_labels
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels

train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
xgb_train <- xgb.DMatrix(data = data.matrix(train_exp), label = train_labels) 
xgb_test <- xgb.DMatrix(data = data.matrix(test_exp), label = test_labels) 
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,  
  max_depth = 4, 
  subsample = 0.5, 
  colsample_bytree = 0.8,#
  gamma = 0.1,
  min_child_weight = 4,lambda=1,alpha=1,scale_pos_weight = 9  
)  
watchlist <- list(train = xgb_train, test = xgb_test)
set.seed(123)
model <- xgb.train(params = params, data = xgb_train, nrounds = 500, watchlist = watchlist, early_stopping_rounds = 50)
importance_matrix <- xgb.importance(feature_names = colnames(xgb_train), model = model)
print(importance_matrix)
XGBOOST <- importance_matrix[importance_matrix$Gain > 0, ]$Feature
###############
importance_single_column <- data.frame(Feature = importance_matrix$Feature, Importance = importance_matrix$Gain)
importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
rownames(importance_filtered) <- importance_filtered$Feature
importance_filtered <- importance_filtered[ , -1, drop = FALSE]
all_result_importance[[paste0("XGBOOST")]] <- as.data.frame(importance_filtered)
pred.xgb <- predict(model,newdata=xgb_train,type="prob")
roc_xgbtrain <- pROC::roc(train_labels, pred.xgb)
roc_xgbtrain
pred.xgbtest <- predict(model,newdata=xgb_test ,type="prob")
roc_xgbtest <- pROC::roc(test_labels, pred.xgbtest)
roc_xgbtest
optimal_coords <- coords(roc_xgbtrain, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(model,newdata=xgb_train,type="prob"))
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, newdata=xgb_test, type = "prob"))
  tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(test_labels == 1, "positive", "negative"), levels = levels_to_use)
  return(tresult)
}, model = model, labelsdata = all_labels)

all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("XGBOOST")
result_metrics <- sapply(all_result, function(x) {
  x$predict_result <- factor(x$predict_result, levels = levels_to_use)
  x$real_label <- factor(x$real_label, levels = levels_to_use)
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
result_metrics
#BT
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
xgb_train <- xgb.DMatrix(data = train_exp, label = train_labels) 
xgb_test <- xgb.DMatrix(data = test_exp, label = test_labels) 
set.seed(123)
n_bootstrap <- 1000 
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(xgb_train), size = nrow(xgb_train), replace = TRUE)
  boot_train_expd <- train_exp[sample_idx, ]
  boot_labels <- train_labels[sample_idx]
  boot_test_expd <- train_exp[-sample_idx, ]
  boot_test_labels <- train_labels[-sample_idx]
  btxgb_test <- xgb.DMatrix(data = boot_test_expd, label = boot_test_labels) 
  dtrain <- xgb.DMatrix(data = boot_train_expd , label = boot_labels)
  watchlist <- list(train = dtrain , test = btxgb_test)
  model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0,
                     watchlist = watchlist, early_stopping_rounds = 50)
  pred_probs <- predict(model, newdata = boot_test_expd,type="prob")
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7 
mccv_results <- lapply(1:n_mccv, function(i) {
  train_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
  mccv_train_expd <- train_exp[train_idx, ]
  mccv_train_labels <- train_labels[train_idx]
  mccv_test_expd <- train_exp[-train_idx, ]
  mccv_test_labels <- train_labels[-train_idx]
  
  mccvxgb_test <- xgb.DMatrix(data = mccv_test_expd, label = mccv_test_labels) 
  dtrain <- xgb.DMatrix(data = mccv_train_expd , label = mccv_train_labels)
  watchlist <- list(train = dtrain , test = mccvxgb_test)
  model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0,watchlist = watchlist
                     , early_stopping_rounds = 50)
  
  pred_probs <- predict(model, newdata = mccv_test_expd ,type="prob")
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
mccv_avg_metrics
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#cv
set.seed(123)
folds <- createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
cv_results <- list()
for (i in seq_along(folds)) {
  train_idx <- folds[[i]]
  train_data <- train_exp[train_idx, ]
  train_labels_cv <- train_labels[train_idx]
  CV_data <- train_exp[-train_idx, ]
  CV_labels <- train_labels[-train_idx]
  dtrain <- xgb.DMatrix(data = train_data , label = train_labels_cv)
  dtest <- xgb.DMatrix(data = CV_data , label = CV_labels)
  watchlist <- list(train = dtrain , test = dtest)
  model <- xgb.train(params = params, data = dtrain, nrounds = 500, watchlist = watchlist,
                     early_stopping_rounds = 50)
  predictions_cv <- predict(model, newdata = CV_data, type = "vector")
  roc_curve_cv <- pROC::roc(CV_labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(CV_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve_cv)
  cv_results[[i]] <- c(
    accuracy = as.numeric(accuracy_cv), 
    recall = as.numeric(recall_cv), 
    precision = as.numeric(precision_cv), 
    f1_score = as.numeric(f1_score_cv), 
    npv = as.numeric(npv_cv), 
    ppv = as.numeric(ppv_cv), 
    auc = as.numeric(auc_value)
  )
}
cv_results_df <- do.call(rbind, cv_results)
mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
cv_results <- list()
cv_metrics <- c(
  accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
  f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics


##############################################################################
#######################################' 10. Support vector machine
##############################################################################

########
Linear##
########
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
fitControl <- trainControl(
  method = "cv",    
  number = fold,    
  classProbs = TRUE,  
  summaryFunction = twoClassSummary,  
  verboseIter = TRUE)
param_grid <- expand.grid(
  C = seq(0.01, 2, 0.01))
tuned_model <- caret::train(
  labels ~ .,                  
  data = train_expd,
  method = "svmLinear",  
  tuneGrid = param_grid,  
  trControl = fitControl)

print(tuned_model)

train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
test_expd <- as.data.frame(test_exp)
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
param_grid <- expand.grid(
  C = 0.02)
set.seed(123)
model <- caret::train(
  labels ~ .,                
  data = train_expd,
  method = "svmLinear",
  tuneGrid = param_grid,
  trControl = fitControl)

pred.svm <- predict(model,newdata=train_expd,type="prob")
roc_curve <- pROC::roc(train_expd$labels, pred.svm[,2])
roc_curve

pred.svmtest <- predict(model,newdata=test_expd,type="prob")
roc_curvetest <- pROC::roc(test_labels, pred.svmtest[, 2])
roc_curvetest

importance <- varImp(model)
print(importance)

plot(importance)
importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
rownames(importance_filtered) <- importance_filtered$Feature
importance_filtered <- importance_filtered[ , -1, drop = FALSE]
all_result_importance[[paste0("SVM")]] <- as.data.frame(importance_filtered)
SVM<- rownames(importance_filtered)
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold

train_result <- as.data.frame(predict(model, type="prob"))
train_result$type <- factor(ifelse(train_result[,2] > cuti, "positive", "negative") )
train_result <- train_result[,-1]
colnames(train_result) <- c("predict_p", "predict_result")
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, expdata, type = "prob"))
  tresult$type <- factor(ifelse(tresult[, 2] > cuti, "positive", "negative"), levels = levels_to_use)
  tresult <- tresult[,-1]
  colnames(tresult) <- c("predict_p", "predict_result")
  tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result

result_label <- paste0("SVM")
result_metrics <- sapply(all_result, function(x) {
  x$predict_result <- factor(x$predict_result, levels = c("negative", "positive"))
  x$real_label <- factor(x$real_label, levels = c("negative", "positive"))

  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p), levels = c("negative", "positive"), direction = "<"))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
#BT
set.seed(123)
n_bootstrap <- 100  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_expd$labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_expd$labels[-sample_idx]
  model <- caret::train(
    labels ~ .,                  
    data = boot_train_expd,
    method = "svmLinear",
    tuneGrid = param_grid,
    trControl = fitControl)
  pred_probs <- predict(model, newdata = boot_test_expd,type="prob")[,2]
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  levels(boot_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- boot_test_labels
  pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- mccv_train_expd$labels
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- mccv_test_expd$labels
  model <- caret::train(
    labels ~ .,                   
    data = mccv_train_expd,
    method = "svmLinear",
    tuneGrid = param_grid,
    trControl = fitControl)
  pred_probs <- predict(model, newdata = mccv_test_expd,type="prob")[,2]
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- mccv_test_labels
  pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")

  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#cv
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
set.seed(123)
model_cv <- caret::train(
  labels ~ .,                    
  data = train_expd,
  method = "svmLinear",
  tuneGrid = param_grid,
  trControl = train_control)
resample_results <- model_cv$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
importance <- varImp(model_cv)
predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv)
optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
cuti_cv <- optimal_coords_cv$threshold

pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")

accuracy_cv <- conf_matrix_cv$overall['Accuracy']
recall_cv <- conf_matrix_cv$byClass['Recall']
precision_cv <- conf_matrix_cv$byClass['Precision']
f1_score_cv <- conf_matrix_cv$byClass['F1']
npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
cv_metrics <- c(
  accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
  f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics

all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics


##############################################################################
#######################################' 11. Grandient Boosting Machine
##############################################################################
train_expd <- as.data.frame(train_exp)
train_expd$labels <- as.character(train_labels)
train_expd$labels <- as.numeric(train_expd$labels)
###############################
set.seed(123)
model <- gbm(
  labels ~ . - labels, 
  data = train_expd, 
  distribution = "bernoulli",    
  n.trees = 300,                
  interaction.depth = 5,        
  shrinkage = 0.01,             
  n.minobsinnode = 10,            
  bag.fraction = 0.5,           
  train.fraction = 1,          
  verbose = TRUE                 
)
all_result_importance[[paste0("GBM")]] <- as.data.frame(summary.gbm(model))[,-1,drop=F]
pred.GBM <- predict(model,newdata=train_expd,type="response")
roc_curve <- pROC::roc(train_expd$labels, pred.GBM)
roc_curve
pred.GBMtest <- predict(model, newdata = test_expd, type = "response")
roc_GBMtest <- pROC::roc(test_labels, pred.GBMtest)
roc_GBMtest
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
importance <- summary(model)
print(importance)
importance_data <- as.data.frame(summary(model, plotit = FALSE))
non_zero_features <- importance_data[importance_data$rel.inf > 0, ]
GBM <- rownames(non_zero_features)
train_result <- as.data.frame(predict.gbm(model, as.data.frame(train_exp),type = "response"))  
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result","real_label")
all_result <- lapply(test_explist, function(data, model, labelsdata){
  data = as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd,2]
  expdata <- data[comd,]
  tresult <- as.data.frame(predict(model, as.data.frame(expdata),type = "response"))
  tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result","real_label")
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("GBM")
result_metrics <- sapply(all_result, function(x) {
  x$predict_result <- factor(x$predict_result, levels = levels_to_use)
  x$real_label <- factor(x$real_label, levels = levels_to_use)
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
result_metrics
#BT
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_expd$labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_expd$labels[-sample_idx]
  model <- gbm(
    labels ~ . - labels, 
    data = boot_train_expd, 
    distribution = "bernoulli",  
    n.trees = 300,              
    interaction.depth = 5,        
    shrinkage = 0.01,            
    n.minobsinnode = 10,           
    bag.fraction = 0.5,          
    train.fraction = 1,                  
    verbose = TRUE           
  )
  pred_probs <- predict(model, newdata = boot_test_expd,type="response")
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  levels(boot_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']

  auc_value <- auc(roc_curve)

  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})

bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
set.seed(123)
n_mccv <- 100  #
train_ratio <- 0.7 
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_test_expd <- train_expd[-sample_idx, ]
  model <- gbm(
    labels ~ . - labels, 
    data = mccv_train_expd, 
    distribution = "bernoulli",    
    n.trees = 300,                
    interaction.depth = 5,         
    shrinkage = 0.01,              
    n.minobsinnode = 10,            
    bag.fraction = 0.5,            
    train.fraction = 1,        
    verbose = TRUE           
  )

  pred_probs <- predict(model, newdata = mccv_test_expd,type="response")
  roc_curve <- pROC::roc(mccv_test_expd$labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  mccv_test_labels <- factor(ifelse(mccv_test_expd$labels==1, "positive", "negative"))
  levels(mccv_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- mccv_test_labels
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})

mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
result_metrics 
#CV
set.seed(123)
folds <- createFolds(train_labels, k = 10, returnTrain = TRUE)
cv_results <- list()
for (i in seq_along(folds)) {
  train_idx <- folds[[i]]
  train_data <- train_expd[train_idx, ]
  CV_labels <- train_labels[train_idx]
  test_data <- train_expd[-train_idx, ]
  test_labels_cv <- train_labels[-train_idx]
  model <- gbm(
    labels ~ . - labels, 
    data = train_data, 
    distribution = "bernoulli",   
    n.trees = 300,               
    interaction.depth = 5,         
    shrinkage = 0.01,              
    n.minobsinnode = 10,           
    bag.fraction = 0.5,            
    train.fraction = 1,          
    verbose = TRUE                #
  )
  predictions_cv <- predict(model, newdata = test_data, type = "response")
  roc_curve_cv <- pROC::roc(test_labels_cv, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(test_labels_cv == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve_cv)
  cv_results[[i]] <- c(
    accuracy = as.numeric(accuracy_cv), 
    recall = as.numeric(recall_cv), 
    precision = as.numeric(precision_cv), 
    f1_score = as.numeric(f1_score_cv), 
    npv = as.numeric(npv_cv), 
    ppv = as.numeric(ppv_cv), 
    auc = as.numeric(auc_value)
  )
}
cv_results_df <- do.call(rbind, cv_results)
mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
cv_results <- list()
cv_metrics <- c(
  accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
  f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
all_result_summary[[result_label]] <- all_result
result_metrics
##############################################################################
#######################################'12.stepwise+Logistic Regression
##############################################################################
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
test_expd_df <- as.data.frame(test_exp)
fullModel = glm(labels~.-labels, family = 'binomial', data = train_expd)
nullModel = glm(labels ~ 1, family = 'binomial', data = train_expd) 
model <- stepAIC(nullModel, 
                 direction = 'both', 
                 scope = list(upper = fullModel, 
                              lower = nullModel), 
                 trace = 0) 
model_formula <- formula(model)
StepWiseAIC <- all.vars(model_formula)[-1]
##
pred.step <- predict(model,newdata=train_expd,type="response")
roc_curve <- pROC::roc(train_expd$labels, pred.step)
roc_curve
pred.steptest <- predict(model,newdata=test_expd_df,type="response")
roc_curvetest <- pROC::roc(test_labels, pred.steptest)
roc_curvetest
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(model, as.data.frame(train_exp),type = "response"))  
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result","real_label")
all_result <- lapply(test_explist, function(data, model, labelsdata){
  data = as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd,2]
  expdata <- data[comd,]
  tresult <- as.data.frame(predict(model, as.data.frame(expdata),type = "response"))
  tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
  tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result","real_label")
  return(tresult)
}, model=model, labelsdata=all_labels)

all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("StepWise-AIC+LR")
result_metrics <- sapply(all_result, function(x) {

  x$predict_result <- factor(x$predict_result, levels = c("negative","positive"))
  x$real_label <- factor(x$real_label, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
result_metrics
#BT
set.seed(123)
n_bootstrap <- 1000  
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_expd$labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_expd$labels[-sample_idx]
  boot_train_expd <- boot_train_expd[, c(StepWiseAIC, "labels")]
  model <- glm(labels ~ ., family = "binomial",data = boot_train_expd)
  pred_probs <- predict(model, newdata = boot_test_expd,type="response")
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  levels(boot_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})

bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
set.seed(123)
n_mccv <- 100 
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- mccv_train_expd$labels
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- mccv_test_expd$labels
  mccv_train_expd <- mccv_train_expd[, c(StepWiseAIC, "labels")]
  model <- glm(labels ~ ., family = "binomial",data = mccv_train_expd)
  pred_probs <- predict(model, newdata = mccv_test_expd,type="response")
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
  levels(mccv_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- mccv_test_labels
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#CV
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
set.seed(123)
model_data <- train_expd[, c(StepWiseAIC, "labels")]
model_cv <- caret::train(
  labels ~ .,
  data = model_data,
  method = "glm",                  
  family = "binomial",             
  metric = "ROC",                   
  trControl = train_control         
)
resample_results <- model_cv$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
importance <- varImp(model_cv)
non_zero_importance <- as.data.frame(importance$importance)
non_zero_importance <- non_zero_importance[non_zero_importance$Overall > 0, , drop = FALSE]  # 筛选重要性大于0的特征
all_result_importance[[paste0("StepWise-AIC+LR")]] <- non_zero_importance
predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv)
optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
cuti_cv <- optimal_coords_cv$threshold
pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
accuracy_cv <- conf_matrix_cv$overall['Accuracy']
recall_cv <- conf_matrix_cv$byClass['Recall']
precision_cv <- conf_matrix_cv$byClass['Precision']
f1_score_cv <- conf_matrix_cv$byClass['F1']
npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
cv_metrics <- c(
  accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
  f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
)
result_metrics <- cbind(cv_metrics, result_metrics)
rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics
##############################################################################
#######################################'13. Naive Bayesian algorithm
##############################################################################
levels_to_use <- c("negative", "positive")  
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
set.seed(123)
model <- naiveBayes(labels ~ ., data = train_expd) 
train_probs <- predict(model,newdata=train_expd, type = "raw")
roc_curve <- roc(train_labels, train_probs[,2])
roc_curve
test_probstest <- predict(model,newdata=test_expd, type = "raw")
roc_curvetest <- roc(test_labels, test_probstest[,2])
roc_curvetest
optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
cuti <- optimal_coords$threshold
train_result <- as.data.frame(predict(model, as.data.frame(train_exp),type = "raw")[,2])  
train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
colnames(train_result) <- c("predict_p", "predict_result","real_label")
all_result <- lapply(test_explist, function(data, model, labelsdata) {
  data <- as.data.frame(data)
  comd <- intersect(rownames(data), rownames(labelsdata))
  labelsdata <- labelsdata[comd, 2]
  expdata <- data[comd, ]
  tresult <- as.data.frame(predict(model, as.data.frame(expdata),type = "raw")[,2])  
  tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
  tresult$real_label <- factor(ifelse(test_labels==1, "positive", "negative"))
  colnames(tresult) <- c("predict_p", "predict_result","real_label")
  return(tresult)
}, model = model, labelsdata = all_labels)
all_result[[length(all_result) + 1]] <- train_result
result_label <- paste0("NaiveBayes")
result_metrics <- sapply(all_result, function(x) {

  x$predict_result <- factor(x$predict_result, levels = levels_to_use)
  x$real_label <- factor(x$real_label, levels = levels_to_use)
  conf_matrix <- confusionMatrix(x$real_label, x$predict_result,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})
#BT
set.seed(123)
n_bootstrap <- 1000 
bootstrap_results <- lapply(1:n_bootstrap, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
  boot_train_expd <- train_expd[sample_idx, ]
  boot_labels <- train_expd$labels[sample_idx]
  boot_test_expd <- train_expd[-sample_idx, ]
  boot_test_labels <- train_expd$labels[-sample_idx]
  model <- naiveBayes(labels ~ ., data = boot_train_expd)
  pred_probs <- predict(model, newdata = boot_test_expd,type="raw")[,2]
  roc_curve <- pROC::roc(boot_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
  pred_labels <- factor(pred_labels, levels = c("negative","positive"))
  conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})

bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
#mccv
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
set.seed(123)
n_mccv <- 100  
train_ratio <- 0.7  
mccv_results <- lapply(1:n_mccv, function(i) {
  sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
  mccv_train_expd <- train_expd[sample_idx, ]
  mccv_train_labels <- mccv_train_expd$labels
  mccv_test_expd <- train_expd[-sample_idx, ]
  mccv_test_labels <- mccv_test_expd$labels
  model <- naiveBayes(labels ~ ., data = mccv_train_expd) 
  pred_probs <- predict(model, newdata = mccv_test_expd,type="raw")[,2]
  roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold

  mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
  levels(mccv_test_labels)
  pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
  real_labels <- mccv_test_labels
  
  pred_labels <- factor(pred_labels, levels = levels_to_use)
  conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")

  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  auc_value <- auc(roc_curve)
  metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
  return(metrics)
})

mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
result_metrics <- cbind(mccv_avg_metrics, result_metrics)
#CV
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
levels(train_expd$labels)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
set.seed(123)
model_cv <- caret::train(labels ~ .,                   
                         data = train_expd, 
                         method = "nb",            
                         trControl = train_control,  
                         metric = "ROC")           
resample_results <- model_cv$resample
mean(resample_results$ROC) 
auc_value <- mean(resample_results$ROC)
importance <- varImp(model_cv)
importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
rownames(importance_filtered) <- importance_filtered$Feature
importance_filtered <- importance_filtered[ , -1, drop = FALSE]
all_result_importance[[paste0("NaiveBayes")]] <- as.data.frame(importance_filtered)

NB <- rownames(importance_filtered)
predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv)
optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
cuti_cv <- optimal_coords_cv$threshold
pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
conf_matrix_cv <- confusionMatrix(real_labels_cv, pred_labels_cv,positive = "positive")
accuracy_cv <- conf_matrix_cv$overall['Accuracy']
recall_cv <- conf_matrix_cv$byClass['Recall']
precision_cv <- conf_matrix_cv$byClass['Precision']
f1_score_cv <- conf_matrix_cv$byClass['F1']
npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
cv_metrics <- c(
  accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
  f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
)
result_metrics <- cbind(cv_metrics, result_metrics)

rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
result_metrics <- result_metrics[, -c(4, 5)]
result_metrics
all_result_acc[[result_label]] <- result_metrics['accuracy', ]
all_result_recall[[result_label]] <- result_metrics['recall', ]
all_result_FS[[result_label]] <- result_metrics['f1_score', ]
all_result_npv[[result_label]] <- result_metrics['npv', ]
all_result_ppv[[result_label]] <- result_metrics['ppv', ]
all_result_pre[[result_label]] <- result_metrics['precision', ]
all_result_auc[[result_label]] <- result_metrics['auc', ]  
all_result_summary[[result_label]] <- all_result
result_metrics

################################################
4X10=40

feature_sets <- list(
  lasso_lambda.1se = lasso1se,
  lasso_lambda.min = lassomin,
  ENR_lambda.min = ENRmin,
  ENR_lambda.1se = ENR1se
)
##############################################################################
#######################################' 2.1 LR
##############################################################################
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  set.seed(123)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  model <- glm(labels ~ ., family = "binomial", data = train_expd)
  importancei <- as.data.frame(model$coefficients[-1])
  colnames(importancei) <- "coefficients"
  all_result_importance[[paste0(feature_name," & LR")]] <- importancei
  train_probs <- predict(model, type = "response")
  roc_curve <- roc(train_labels, train_probs, levels = c(0, 1), direction = "<")
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  levels_to_use <- c("negative","positive")
  train_probs <- predict(model, type = "response")
  roc_curve <- roc(train_labels, train_probs, levels = c(0, 1), direction = "<")
  auc_value <- auc(roc_curve)
  train_result <- as.data.frame(predict(model, type="response"))
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "response"))
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = c("negative","positive"))
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = c("negative","positive"))
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & LR")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative","positive"))
    x$real_label <- factor(x$real_label, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  # Bootstrap 
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_labels[-sample_idx]
    model <- glm(labels ~ ., family = "binomial", data = boot_train_expd)
    pred_boot <- predict(model, newdata = boot_test_expd, type = "response")
    roc_curve <- pROC::roc(boot_test_labels, pred_boot)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
    real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
    pred_labels_boot <- factor(pred_labels_boot, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  # MCCV 
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- train_labels[sample_idx]
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- train_labels[-sample_idx]
    model <- glm(labels ~ ., family = "binomial", data = mccv_train_expd)
    pred_mccv <- predict(model, newdata = mccv_test_expd, type = "response")
    roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
    real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
    pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  # CV
  train_expd$labels <- factor(train_labels)
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_cv <- caret::train(
    labels ~ ., data = train_expd, 
    method = "glm",
    family = "binomial",
    metric = "ROC",  
    trControl = train_control
  )
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
  all_result_summary[[result_label]] <- all_result
  result_metrics
}

##############################################################################
#######################################' 2.2  Linear discriminant analysis
##############################################################################
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  model <- lda(labels ~ ., data = train_expd)
  all_result_importance[[paste0(feature_name," & LDA")]] <- as.data.frame(model$scaling)
  train_predict <- predict(model, train_expd)
  train_probs <- train_predict$posterior[,2]  
  roc_curve <- roc(train_expd$labels, train_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  roc_curve <- roc(train_labels, train_probs)
  auc_value <- auc(roc_curve)
  train_result <- as.data.frame(train_probs)
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)[, selected_features]
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata)$posterior[, 2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & LDA")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p),levels = c("negative", "positive"), direction = "<"))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BOOSTRAP
  set.seed(123)
  n_bootstrap <- 1000 
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_labels[-sample_idx]
    model <- lda(labels ~ ., data = boot_train_expd)
    pred_probs <- predict(model, newdata = boot_test_expd)$posterior[, 2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs,levels = c(0, 1), direction = "<")
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    boot_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    levels(boot_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels ,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })

  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- lda(labels ~ ., data = mccv_train_expd)
    pred_probs <- predict(model, newdata = mccv_test_expd)$posterior[, 2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs,levels = c(0, 1), direction = "<")
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_cv <- caret::train(
    labels ~ ., data = train_expd, 
    method = "lda",
    metric = "ROC",  
    trControl = train_control
  )
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv,levels = c("negative", "positive"), direction = "<")
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}

##############################################################################
#######################################' 2.3  k-Nearest Neighbor
##############################################################################
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "cv", number = 10)  
  model <- caret::train(
    labels ~ .,
    data = train_expd,
    method = "knn",
    trControl = train_control,
    tuneGrid = data.frame(k = 70)
  )
  print(model$bestTune)
  importance <- varImp(model)
  print(importance)
  plot(importance)
  importance <- varImp(model)
  importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & KNN")]] <- as.data.frame(importance_filtered)
  best_k <- model$bestTune$k
  train_probs <- predict(model, type="prob")[,2]
  roc_curve <- roc(train_expd$labels, train_probs)
  roc_curve
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti<- optimal_coords$threshold
  #####
  train_result <- as.data.frame(predict(model, type="prob")[,2])
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)[, selected_features]
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "prob")[,2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & KNN")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative","positive"))
    x$real_label <- factor(x$real_label, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BOOTSTRAP
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- caret::train(
      labels ~ .,
      data = boot_train_expd,
      method = "knn",
      trControl = train_control,
      tuneGrid = data.frame(k =70)
    )
    pred_probs <- predict(model, newdata =  boot_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- boot_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(pred_labels,real_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- caret::train(
      labels ~ .,
      data = mccv_train_expd,
      method = "knn",
      trControl = train_control,
      tuneGrid = data.frame(k = 70)
    )
    pred_probs <- predict(model, newdata = mccv_test_expd, type = "prob")[, 2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"), levels = levels_to_use)
    pred_labels <- factor(ifelse(pred_probs > cuti, "positive", "negative"), levels = levels_to_use)
    conf_matrix <- confusionMatrix(mccv_test_labels,pred_labels, positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)

    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  set.seed(123)
  model <- caret::train(
    labels ~ .,
    data = train_expd,
    method = "knn",
    trControl = train_control,
    tuneGrid = data.frame(k =70)
  )
  resample_results <- model$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
  roc_curve <- roc(train_labels, predictions)
  predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
  roc_curve <- roc(train_labels, predictions)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  CV_metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  print(CV_metrics)
  result_metrics <- cbind(CV_metrics,result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################' 2.4  Decision tree
##############################################################################
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  set.seed(123)
  control_params <- tree.control(
    nobs = nrow(train_expd),  
    mincut = 7,                
    minsize = 40,             
    mindev = 0.01      
  )
  model <- tree(labels ~ . - labels, data = train_expd, control = control_params)
  predictions <- predict(model, newdata = train_expd, type = "vector")[,2]
  roc_DT <- pROC::roc(train_labels, predictions)
  roc_DT
  optimal_coords <- coords(roc_DT, "best", best.method = "youden", ret = "all")
  cuti<- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, type="vector"))
  train_result$type <- factor(ifelse(train_result[,2] > cuti, "positive", "negative"))
  train_result <- train_result[,-1]
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "vector")[,2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result

  result_label <- paste0(feature_name," & DT")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)),levels = c("negative", "positive"),  # 指定类别顺序
                     direction = "<")
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BOOTSTRAP
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- tree(labels ~ . - labels, data = boot_train_expd, control = control_params)
    pred_probs <- predict(model, newdata = boot_test_expd,type="vector")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(pred_labels,real_labels, positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #MCCV
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- tree(labels ~ . - labels, data = mccv_train_expd, control = control_params)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="vector")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  
  
  #cv
  set.seed(123)
  folds <- createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    train_test_data <- train_expd[-train_idx, ]
    model <- tree(labels ~ . - labels, data = train_data, control = control_params)
    predictions_cv <- predict(model, newdata = train_test_data, type = "vector")[,2]
    roc_curve_cv <- pROC::roc(train_test_data$labels, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(train_test_data$labels == 1, "positive", "negative"))
    #real_labels_cv <- train_test_data$labels
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
  all_result_summary[[result_label]] <- all_result
  result_metrics
  importance_values <- numeric(length(used_features))
  names(importance_values) <- used_features
  non_leaf_nodes <- which(model$frame$var != "<leaf>")
  for (i in non_leaf_nodes) {
    feature <- as.character(model$frame$var[i])
    dev_reduction <- model$frame$dev[i] - sum(model$frame$dev[model$children[[i]]])
    importance_values[feature] <- importance_values[feature] + dev_reduction
  }
  importance_df <- data.frame(
    Overall = importance_values,
    row.names = names(importance_values)
  ) %>% 
    arrange(-Overall) %>% 
    filter(Overall > 0)
  all_result_importance[[paste0(feature_name, " & DT")]] <- importance_df
  print(importance_df)
}



##############################################################################
#######################################' 2.5 Random forest
##############################################################################
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  test_expd <- as.matrix(test_exp)[, selected_features]
  testlabels <- factor(test_labels)
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  ctrl <- trainControl(method = "cv", number = 2,classProbs = TRUE,    
                       summaryFunction = twoClassSummary, 
                       verboseIter = TRUE)
  grid <- expand.grid(mtry = 7)
  set.seed(123)
  weights <- ifelse(train_expd$labels == "positive", 8, 1)
  model <- caret::train(labels ~ .,
                        data = train_expd,
                        method = "rf",   
                        trControl = ctrl,
                        tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
  pred.RF <- predict(model,newdata=train_expd,type="prob")
  pred.RF <- predict(model,newdata=train_expd,type="prob")
  roc_RFtrain <- pROC::roc(train_expd$labels, pred.RF[, 2])
  roc_RFtrain
  pred.RFtest <- predict(model,newdata=test_expd,type="prob")
  roc_RFtest <- pROC::roc(testlabels, pred.RFtest[, 2])
  roc_RFtest
  importance <- varImp(model)
  plot(importance)
  print(importance)
  importance <- varImp(model)  
  non_zero_importance <- as.data.frame(importance$importance)
  non_zero_importance <- non_zero_importance[non_zero_importance$Overall > 0, , drop = FALSE] 
  all_result_importance[[paste0(feature_name," & RF")]] <- non_zero_importance
  train_probs <- predict(model, type="prob")[,2]
  roc_curve <- roc(train_expd$labels, train_probs)
  roc_curve
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti<- optimal_coords$threshold
  #####
  train_result <- as.data.frame(predict(model, type="prob")[,2])
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "prob")[,2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & RF")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BT
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  set.seed(123)
  n_bootstrap <- 1000 
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- boot_test_expd$labels
    weights <- ifelse(boot_train_expd$labels == "positive", 8, 1)
    model <- caret::train(labels ~ .,
                          data = boot_train_expd,
                          method = "rf",  
                          trControl = ctrl,
                          tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)

    pred_probs <- predict(model, newdata = boot_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- boot_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7 
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    weights <- ifelse(mccv_train_expd$labels == "positive", 8, 1)
    model <- caret::train(labels ~ .,
                          data = mccv_train_expd,
                          method = "rf",  
                          trControl = ctrl,
                          tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })

  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  folds <- createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    train_labels_CV <- train_labels[train_idx]
    test_data <- train_expd[-train_idx, ]
    CV_labels <- train_labels[-train_idx]
    weights <- ifelse(train_data$labels == "positive", 8, 1)
    set.seed(123)
    model <- caret::train(labels ~ .,
                          data = train_data,
                          method = "rf",  
                          trControl = ctrl,
                          tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)

    predictions_cv <- predict(model, newdata =  test_data, type = "prob")[,2]
    roc_curve_cv <- pROC::roc(CV_labels, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(CV_labels == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ] 
  all_result_summary[[result_label]] <- all_result
  result_metrics
}

##############################################################################
#######################################' 2.6 XGBOOST
##############################################################################
all_labels <- bind_rows(labels_list)
rownames(all_labels) <- all_labels[,1]
all_labels[,2] <- ifelse(all_labels[,2]==type, 1, 0)
train_exp <- exp_list[[which(exp_file==train_exp_names)]]
test_explist <- exp_list[which(!exp_file==train_exp_names)]
com <- intersect(rownames(train_exp), rownames(all_labels))
train_labels <- all_labels[com,]
train_exp <-train_exp[com,]
train_labels <- train_labels[, 2]

test_exp_names <- "DatasetB.txt"
test_exp <- exp_list[[which(exp_file==test_exp_names)]]
comtest <- intersect(rownames(test_exp), rownames(all_labels))
test_labels <- all_labels[comtest,]
test_exp <-test_exp[comtest,]
test_labels <- test_labels[, 2]

test_expd <- as.data.frame(test_exp)
test_expd$labels <- test_labels
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]

  train_expd <- as.data.frame(train_exp[, selected_features])
  test_expd <- as.data.frame(test_exp[, selected_features])
  xgb_train <- xgb.DMatrix(data = data.matrix(train_expd), label = train_labels, weight = ifelse(train_labels == 1, 5, 1)) 
  xgb_test <- xgb.DMatrix(data = data.matrix(test_expd), label = test_labels) 
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1,  
    max_depth = 4, 
    subsample = 0.5, 
    colsample_bytree = 0.8,
    gamma = 0.1,
    min_child_weight = 4,lambda=1,alpha=1,scale_pos_weight = 9
  )
  watchlist <- list(train = xgb_train, test = xgb_test)
  set.seed(123) 
  model <- xgb.train(params = params, data = xgb_train, nrounds = 500, watchlist = watchlist, early_stopping_rounds = 50)
  importance_matrix <- xgb.importance(feature_names = colnames(xgb_train), model = model)
  print(importance_matrix)
  importance_single_column <- data.frame(Feature = importance_matrix$Feature, Importance = importance_matrix$Gain)
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & XGBOOST")]] <- as.data.frame(importance_filtered)
  pred.xgb <- predict(model,newdata=xgb_train,type="prob")
  roc_xgbtrain <- pROC::roc(train_labels, pred.xgb)
  roc_xgbtrain
  pred.xgbtest <- predict(model,newdata=xgb_test ,type="prob")
  roc_xgbtest <- pROC::roc(test_labels, pred.xgbtest)
  roc_xgbtest
  optimal_coords <- coords(roc_xgbtrain, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model,newdata=xgb_train,type="prob"))
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, newdata=xgb_test, type = "prob"))
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(test_labels == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & XGBOOST")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BT
  set.seed(123)
  n_bootstrap <- 1000 
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(xgb_train), size = nrow(xgb_train), replace = TRUE)
    boot_train_expd <- train_exp[sample_idx, selected_features]
    boot_labels <- train_labels[sample_idx]
    boot_test_expd <- train_exp[-sample_idx, selected_features]
    boot_test_labels <- train_labels[-sample_idx]
    btxgb_test <- xgb.DMatrix(data = boot_test_expd, label = boot_test_labels) 
    dtrain <- xgb.DMatrix(data = boot_train_expd , label = boot_labels)
    watchlist <- list(train = dtrain , test = btxgb_test)
    model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0,
                       watchlist = watchlist, early_stopping_rounds = 50)
    pred_probs <- predict(model, newdata = btxgb_test,type="prob")
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics<- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_exp[sample_idx, selected_features]
    mccv_train_labels <- train_labels[sample_idx]
    mccv_test_expd <- train_exp[-sample_idx, selected_features]
    mccv_test_labels <- train_labels[-sample_idx]
    mccvxgb_test <- xgb.DMatrix(data = mccv_test_expd, label = mccv_test_labels) 
    dtrain <- xgb.DMatrix(data = mccv_train_expd , label = mccv_train_labels)
    watchlist <- list(train = dtrain , test = mccvxgb_test)
    model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0,watchlist = watchlist
                       , early_stopping_rounds = 50)
    
    pred_probs <- predict(model, newdata = mccv_test_expd ,type="prob")
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(pred_labels,real_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  folds <- createFolds(train_labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_exp[train_idx, selected_features]
    train_labels_cv <- train_labels[train_idx]
    CV_data <- train_exp[-train_idx,selected_features ]
    CV_labels <- train_labels[-train_idx]
    dtrain <- xgb.DMatrix(data = train_data , label = train_labels_cv)
    dtest <- xgb.DMatrix(data = CV_data , label = CV_labels)
    watchlist <- list(train = dtrain , test = dtest)
    set.seed(123)
    model <- xgb.train(params = params, data = dtrain, nrounds = 500, watchlist = watchlist,
                       early_stopping_rounds = 50)
    predictions_cv <- predict(model, newdata = CV_data, type = "vector")
    roc_curve_cv <- pROC::roc(CV_labels, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(CV_labels == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}

##############################################################################
#######################################' 2.7 Support vector machine
##############################################################################
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  test_expd <- as.data.frame(test_exp[, selected_features])
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  param_grid <- expand.grid(
    C = 0.02)
  set.seed(123)
  model <- caret::train(
    labels ~ .,                   
    data = train_expd,
    method = "svmLinear",
    tuneGrid = param_grid,
    trControl = fitControl)
  pred.svm <- predict(model,newdata=train_expd,type="prob")
  roc_curve <- pROC::roc(train_expd$labels, pred.svm[,2])
  roc_curve
  pred.svmtest <- predict(model,newdata=test_expd,type="prob")
  roc_curvetest <- pROC::roc(test_labels, pred.svmtest[, 2])
  roc_curvetest
  importance <- varImp(model)
  print(importance)
  plot(importance)
  importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & SVM")]] <- as.data.frame(importance_filtered)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, type="prob"))
  train_result$type <- factor(ifelse(train_result[,2] > cuti, "positive", "negative") )
  train_result <- train_result[,-1]
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "prob"))
    tresult$type <- factor(ifelse(tresult[, 2] > cuti, "positive", "negative"), levels = levels_to_use)
    tresult <- tresult[,-1]
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & SVM")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative", "positive"))
    x$real_label <- factor(x$real_label, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p), levels = c("negative", "positive"), direction = "<"))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
    
    model <- caret::train(
      labels ~ .,                   
      data = boot_train_expd,
      method = "svmLinear",
      tuneGrid = param_grid,
      trControl = train_control)
    pred_probs <- predict(model, newdata = boot_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- boot_test_labels
    pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })

  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
    
    model <- caret::train(
      labels ~ .,                   
      data = mccv_train_expd,
      method = "svmLinear",
      tuneGrid = param_grid,
      trControl = train_control)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  set.seed(123)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  folds <- createFolds(train_labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    CV_labels <- train_labels[train_idx]
    test_data <- train_expd[-train_idx, ]
    test_labels_cv <- train_labels[-train_idx]
    train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
    model <- caret::train(
      labels ~ .,                  
      data = train_data,
      method = "svmLinear",
      tuneGrid = param_grid, metric = "ROC",
      trControl = train_control)
    predictions_cv <- predict(model, newdata = test_data, type = "prob")[,2]
    roc_curve_cv <- pROC::roc(test_labels_cv, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(test_labels_cv == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################' 2.8  Grandient Boosting Machine
##############################################################################
all_labels <- bind_rows(labels_list)
rownames(all_labels) <- all_labels[,1]
all_labels[,2] <- ifelse(all_labels[,2]==type, 1, 0)
train_exp <- exp_list[[which(exp_file==train_exp_names)]]
test_explist <- exp_list[which(!exp_file==train_exp_names)]
com <- intersect(rownames(train_exp), rownames(all_labels))
train_labels <- all_labels[com,]
train_exp <-train_exp[com,]
train_labels <- train_labels[, 2]

test_exp_names <- "DatasetB.txt"
test_exp <- exp_list[[which(exp_file==test_exp_names)]]
comtest <- intersect(rownames(test_exp), rownames(all_labels))
test_labels <- all_labels[comtest,]
test_exp <-test_exp[comtest,]
test_labels <- test_labels[, 2]

test_expd <- as.data.frame(test_exp)
test_expd$labels <- test_labels
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  test_expd <- as.data.frame(test_exp[, selected_features])
  train_expd$labels <- as.numeric(train_expd$labels)
  set.seed(123)
  model <- gbm(
    labels ~ . - labels, 
    data = train_expd, 
    distribution = "bernoulli",   
    n.trees = 300,                
    interaction.depth = 5,         
    shrinkage = 0.01,              
    n.minobsinnode = 10,            
    bag.fraction = 0.5,            
    train.fraction = 1,         
    verbose = TRUE
  )
  all_result_importance[[paste0(feature_name," & GBM")]] <- as.data.frame(summary.gbm(model))[,-1,drop=F]
  pred.GBM <- predict(model,newdata=train_expd,type="response")
  roc_curve <- pROC::roc(train_expd$labels, pred.GBM)
  roc_curve
  pred.GBMtest <- predict(model, newdata = test_expd, type = "response")
  roc_GBMtest <- pROC::roc(test_labels, pred.GBMtest)
  roc_GBMtest
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  importance <- summary(model)
  print(importance)
  importance_data <- as.data.frame(summary(model, plotit = FALSE))
  non_zero_features <- importance_data[importance_data$rel.inf > 0, ]
  GBM <- rownames(non_zero_features)
  train_result <- as.data.frame(predict.gbm(model, as.data.frame(train_exp),type = "response"))  
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  colnames(train_result) <- c("predict_p", "predict_result","real_label")
  all_result <- lapply(test_explist, function(data, model, labelsdata){
    data = as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd,2]
    expdata <- data[comd,]
    tresult <- as.data.frame(predict(model, as.data.frame(expdata),type = "response"))
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
    colnames(tresult) <- c("predict_p", "predict_result","real_label")
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & GBM")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- gbm(
      labels ~ . - labels, 
      data = boot_train_expd, 
      distribution = "bernoulli",    
      n.trees = 300,             
      interaction.depth = 5,         
      shrinkage = 0.01,          
      n.minobsinnode = 10,          
      bag.fraction = 0.5,           
      train.fraction = 1,                
      verbose = TRUE             
    )

    pred_probs <- predict(model, newdata = boot_test_expd,type="response")
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })

  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_test_expd <- train_expd[-sample_idx, ]
    model <- gbm(
      labels ~ . - labels, 
      data = mccv_train_expd, 
      distribution = "bernoulli",   
      n.trees = 300,             
      interaction.depth = 5,       
      shrinkage = 0.01,           
      n.minobsinnode = 10,        
      bag.fraction = 0.5,          
      train.fraction = 1,                
      verbose = TRUE                 
    )
    pred_probs <- predict(model, newdata = mccv_test_expd,type="response")
    roc_curve <- pROC::roc(mccv_test_expd$labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_expd$labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  result_metrics 
  #CV
  set.seed(123)
  folds <- caret::createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    CV_labels <- train_labels[train_idx]
    test_datacv <- train_expd[-train_idx, ]
    test_labelscv <- train_labels[-train_idx]
    model <- gbm(
      labels ~ . - labels, 
      data = train_data, 
      distribution = "bernoulli", 
      n.trees = 300,           
      interaction.depth = 5,    
      shrinkage = 0.01,           
      n.minobsinnode = 10,           
      bag.fraction = 0.5,        
      train.fraction = 1,                  
      verbose = TRUE             
    )
    predictions_cv <- predict(model, newdata = test_datacv, type = "response")
    roc_curve_cv <- pROC::roc(test_labelscv, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    cv_results[[i]] <- list(model = model, roc_curve = roc_curve_cv, threshold = cuti_cv)
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(test_labelscv == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################'2.9 stepwise+Logistic Regression
##############################################################################
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  test_expd_df <- as.data.frame(test_exp[, selected_features])
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  fullModel = glm(labels~.-labels, family = 'binomial', data = train_expd) 
  nullModel = glm(labels ~ 1, family = 'binomial', data = train_expd) 
  model <- stepAIC(nullModel, 
                   direction = 'both',
                   scope = list(upper = fullModel, 
                                lower = nullModel),
                   trace = 0) 
  model_formula <- formula(model)
  StepWiseAICCV <- all.vars(model_formula)[-1]
  pred.step <- predict(model,newdata=train_expd,type="response")
  roc_curve <- pROC::roc(train_expd$labels, pred.step)
  roc_curve
  pred.steptest <- predict(model,newdata=test_expd_df,type="response")
  roc_curvetest <- pROC::roc(test_labels, pred.steptest)
  roc_curvetest
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, as.data.frame(train_exp),type = "response"))  
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  colnames(train_result) <- c("predict_p", "predict_result","real_label")
  all_result <- lapply(test_explist, function(data, model, labelsdata){
    data = as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd,2]
    expdata <- data[comd,]
    tresult <- as.data.frame(predict(model, as.data.frame(expdata),type = "response"))
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
    colnames(tresult) <- c("predict_p", "predict_result","real_label")
    return(tresult)
  }, model=model, labelsdata=all_labels)
  
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & StepWise-AIC+LR")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative","positive"))
    x$real_label <- factor(x$real_label, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    boot_train_expd <- boot_train_expd[, c(StepWiseAICCV, "labels")]
    model <- glm(labels ~ ., family = "binomial",data = boot_train_expd)
    pred_probs <- predict(model, newdata = boot_test_expd,type="response")
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })

  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    mccv_train_expd <- mccv_train_expd[, c(StepWiseAICCV, "labels")]
    model <- glm(labels ~ ., family = "binomial",data = mccv_train_expd)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="response")
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_data <- train_expd[, c(StepWiseAICCV, "labels")]
  model_cv <- caret::train(
    labels ~ .,
    data = model_data,
    method = "glm",                  
    family = "binomial",             
    metric = "ROC",                   
    trControl = train_control        
  )
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  importance <- varImp(model_cv)
  non_zero_importance <- as.data.frame(importance$importance)
  non_zero_importance <- non_zero_importance[non_zero_importance$Overall > 0, , drop = FALSE]  # 筛选重要性大于0的特征
  all_result_importance[[paste0(feature_name," & StepWise-AIC+LR")]] <- non_zero_importance
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################'2.10 Naive Bayesian algorithm
##############################################################################
levels_to_use <- c("negative", "positive")
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  set.seed(123)
  model <- naiveBayes(labels ~ ., data = train_expd)
  train_probs <- predict(model,newdata=train_expd, type = "raw")
  roc_curve <- roc(train_labels, train_probs[,2])
  roc_curve
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, as.data.frame(train_exp),type = "raw")[,2])  
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  colnames(train_result) <- c("predict_p", "predict_result","real_label")
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, as.data.frame(test_exp),type = "raw")[,2])  
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    tresult$real_label <- factor(ifelse(test_labels==1, "positive", "negative"))
    colnames(tresult) <- c("predict_p", "predict_result","real_label")
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & NaiveBayes")
  result_metrics <- sapply(all_result, function(x) {
  
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
 
    conf_matrix <- confusionMatrix(x$real_label, x$predict_result,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- naiveBayes(labels ~ ., data = boot_train_expd) 
    pred_probs <- predict(model, newdata = boot_test_expd,type="raw")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- naiveBayes(labels ~ ., data = mccv_train_expd) 

    pred_probs <- predict(model, newdata = mccv_test_expd,type="raw")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)

    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })

  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #CV
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_cv <- caret::train(labels ~ .,                   
                           data = train_expd,  
                           method = "nb",          
                           trControl = train_control, 
                           metric = "ROC")           
  
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  importance <- varImp(model_cv)
  importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & NaiveBayes")]] <- as.data.frame(importance_filtered)
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
  conf_matrix_cv <- confusionMatrix(real_labels_cv, pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ] 
  all_result_summary[[result_label]] <- all_result
  result_metrics
}


##############################################################

train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_data <- train_expd[, -which(names(train_expd) %in% 'labels')]
train_RTW <- train_expd$labels
train_RTW <- as.factor(train_expd$labels)
#
# RFE 
control <- rfeControl(  
  functions = rfFuncs,   
  method = "cv",         
  number = 10
)
sizes <- c(1:50)
set.seed(123)
rfe_results <- rfe(
  x = train_data, 
  y = train_expd$labels,
  sizes = sizes,
  rfeControl = control
)
print(rfe_results)

plot(rfe_results, type = c("g", "o"))
selected_features <- predictors(rfe_results)
RFE <-  selected_features
RFE

###########################
library(caret)
train_expd <- as.data.frame(train_exp)
train_expd$labels <- factor(train_labels)
train_data <- train_expd[, -which(names(train_expd) %in% 'labels')]
train_RTW <- train_expd$labels
train_RTW <- as.factor(train_expd$labels)
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
set.seed(123)
knn_control <- trainControl(method = "cv", number = 10)
rfe_control <- rfeControl(functions = rfFuncs,  
                          method = "cv",
                          number = 10)
knn_rfe <- rfe(labels~ ., 
               data = train_expd, 
               sizes = c(1:50),  
               rfeControl = rfe_control,
               method = "knn", 
               tuneLength = 10,
               tuneGrid = expand.grid(k=70))
# 查看 RFE 选择的最优特征
selected_features <- predictors(knn_rfe)
print(selected_features)
KNN_RFE <- selected_features

###############################################################

rfe_control <- rfeControl(
  functions = ldaFuncs, 
  method = "cv",
  number = 10  
)
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_expd$labels <- as.factor(train_expd$labels)
set.seed(123)
lda_rfe <- rfe(
  x = train_expd[, setdiff(names(train_expd), "labels")],
  y = train_expd$labels, 
  sizes = c(1:50),
  rfeControl = rfe_control
)
selected_features_lda <- predictors(lda_rfe)
print(selected_features_lda)
LDA_RFE <- selected_features_lda
#########################################NB
set.seed(123)
train_expd <- as.data.frame(train_exp)
train_expd$labels <- train_labels
train_labels <- as.factor(train_labels)

rfe_control <- rfeControl(
  functions = nbFuncs,  
  method = "cv",
  number = 10
)
set.seed(123)
train_control <- trainControl(
  method = "cv",
  number = 10,  # 10 折交叉验证
  search = "grid"
)
set.seed(123) 
nb_rfe <- rfe(
  x = train_expd[, -which(names(train_expd) == "labels")], 
  y = train_labels, 
  sizes = c(1:50), 
  rfeControl = rfe_control,
  method = "nb", 
  tuneGrid = expand.grid(
  ),  
  trControl = train_control
)

selected_features_nb <- predictors(nb_rfe)
print(selected_features_nb)
NB_RFE<- selected_features_nb


9X10=90
######################################################################
##############################################################################
#######################################' 3.1 LR
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  #StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  set.seed(123)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  model <- glm(labels ~ ., family = "binomial", data = train_expd)
  importancei <- as.data.frame(model$coefficients[-1])
  colnames(importancei) <- "coefficients"
  all_result_importance[[paste0(feature_name," & LR")]] <- importancei
  train_probs <- predict(model, type = "response")
  roc_curve <- roc(train_labels, train_probs, levels = c(0, 1), direction = "<")
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  levels_to_use <- c("negative","positive")
  train_probs <- predict(model, type = "response")
  roc_curve <- roc(train_labels, train_probs, levels = c(0, 1), direction = "<")
  auc_value <- auc(roc_curve)
  train_result <- as.data.frame(predict(model, type="response"))
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "response"))
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = c("negative","positive"))
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = c("negative","positive"))
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & LR")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative","positive"))
    x$real_label <- factor(x$real_label, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  # Bootstrap 
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_labels[-sample_idx]
    model <- glm(labels ~ ., family = "binomial", data = boot_train_expd)
    pred_boot <- predict(model, newdata = boot_test_expd, type = "response")
    roc_curve <- pROC::roc(boot_test_labels, pred_boot)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels_boot <- ifelse(pred_boot > cuti, "positive", "negative")
    real_labels_boot <- factor(ifelse(boot_test_labels == 1, "positive", "negative"))
    pred_labels_boot <- factor(pred_labels_boot, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels_boot,pred_labels_boot,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  # MCCV 
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = round(nrow(train_expd) * train_ratio))
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- train_labels[sample_idx]
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- train_labels[-sample_idx]
    model <- glm(labels ~ ., family = "binomial", data = mccv_train_expd)
    pred_mccv <- predict(model, newdata = mccv_test_expd, type = "response")
    roc_curve <- pROC::roc(mccv_test_labels, pred_mccv)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels_mccv <- ifelse(pred_mccv > cuti, "positive", "negative")
    real_labels_mccv <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"))
    pred_labels_mccv <- factor(pred_labels_mccv, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels_mccv,pred_labels_mccv,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  # CV
  train_expd$labels <- factor(train_labels)
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_cv <- caret::train(
    labels ~ ., data = train_expd, 
    method = "glm",
    family = "binomial",
    metric = "ROC",  
    trControl = train_control
  )
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################' 3.2 Linear discriminant analysis
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  #LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  model <- lda(labels ~ ., data = train_expd)
  all_result_importance[[paste0(feature_name," & LDA")]] <- as.data.frame(model$scaling)
  train_predict <- predict(model, train_expd)
  train_probs <- train_predict$posterior[,2]  
  roc_curve <- roc(train_expd$labels, train_probs)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  roc_curve <- roc(train_labels, train_probs)
  auc_value <- auc(roc_curve)
  train_result <- as.data.frame(train_probs)
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)[, selected_features]
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata)$posterior[, 2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & LDA")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p),levels = c("negative", "positive"), direction = "<"))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BOOSTRAP
  set.seed(123)
  n_bootstrap <- 1000 
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_labels[-sample_idx]
    model <- lda(labels ~ ., data = boot_train_expd)
    pred_probs <- predict(model, newdata = boot_test_expd)$posterior[, 2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs,levels = c(0, 1), direction = "<")
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    boot_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    levels(boot_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels ,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- lda(labels ~ ., data = mccv_train_expd)
    pred_probs <- predict(model, newdata = mccv_test_expd)$posterior[, 2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs,levels = c(0, 1), direction = "<")
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_cv <- caret::train(
    labels ~ ., data = train_expd, 
    method = "lda",
    metric = "ROC",  
    trControl = train_control
  )
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv,levels = c("negative", "positive"), direction = "<")
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################' 3.3 KNN最邻近 k-Nearest Neighbor
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  #KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "cv", number = 10)  
  model <- caret::train(
    labels ~ .,
    data = train_expd,
    method = "knn",
    trControl = train_control,
    tuneGrid = data.frame(k = 70)
  )
  print(model$bestTune)
  importance <- varImp(model)
  print(importance)
  plot(importance)
  importance <- varImp(model)
  importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & KNN")]] <- as.data.frame(importance_filtered)
  best_k <- model$bestTune$k
  train_probs <- predict(model, type="prob")[,2]
  roc_curve <- roc(train_expd$labels, train_probs)
  roc_curve
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti<- optimal_coords$threshold
  #####
  train_result <- as.data.frame(predict(model, type="prob")[,2])
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)[, selected_features]
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "prob")[,2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & KNN")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative","positive"))
    x$real_label <- factor(x$real_label, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BOOTSTRAP
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- caret::train(
      labels ~ .,
      data = boot_train_expd,
      method = "knn",
      trControl = train_control,
      tuneGrid = data.frame(k =70)
    )
    pred_probs <- predict(model, newdata =  boot_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- boot_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(pred_labels,real_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- caret::train(
      labels ~ .,
      data = mccv_train_expd,
      method = "knn",
      trControl = train_control,
      tuneGrid = data.frame(k = 70)
    )
    pred_probs <- predict(model, newdata = mccv_test_expd, type = "prob")[, 2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels == 1, "positive", "negative"), levels = levels_to_use)
    pred_labels <- factor(ifelse(pred_probs > cuti, "positive", "negative"), levels = levels_to_use)
    conf_matrix <- confusionMatrix(mccv_test_labels,pred_labels, positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  set.seed(123)
  model <- caret::train(
    labels ~ .,
    data = train_expd,
    method = "knn",
    trControl = train_control,
    tuneGrid = data.frame(k =70)
  )
  resample_results <- model$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
  roc_curve <- roc(train_labels, predictions)
  predictions <- predict(model, newdata = train_expd, type = "prob")[,2]
  roc_curve <- roc(train_labels, predictions)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  pred_labels_cv <- ifelse(predictions > cuti, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative", "positive"))
  conf_matrix <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy <- conf_matrix$overall['Accuracy']
  recall <- conf_matrix$byClass['Recall']
  precision <- conf_matrix$byClass['Precision']
  f1_score <- conf_matrix$byClass['F1']
  npv <- conf_matrix$byClass['Neg Pred Value']
  ppv <- conf_matrix$byClass['Pos Pred Value']
  CV_metrics <- c(
    accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, 
    npv = npv, ppv = ppv, auc = auc_value
  )
  print(CV_metrics)
  result_metrics <- cbind(CV_metrics,result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################' 3.4  Decision tree
##############################################################################
feature_sets <- list(
  #DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  set.seed(123)
  control_params <- tree.control(
    nobs = nrow(train_expd),  
    mincut = 7,                
    minsize = 40,             
    mindev = 0.01      
  )
  model <- tree(labels ~ . - labels, data = train_expd, control = control_params)
  predictions <- predict(model, newdata = train_expd, type = "vector")[,2]
  roc_DT <- pROC::roc(train_labels, predictions)
  roc_DT
  optimal_coords <- coords(roc_DT, "best", best.method = "youden", ret = "all")
  cuti<- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, type="vector"))
  train_result$type <- factor(ifelse(train_result[,2] > cuti, "positive", "negative"))
  train_result <- train_result[,-1]
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "vector")[,2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  
  result_label <- paste0(feature_name," & DT")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)),levels = c("negative", "positive"),  # 指定类别顺序
                     direction = "<")
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BOOTSTRAP
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- tree(labels ~ . - labels, data = boot_train_expd, control = control_params)
    pred_probs <- predict(model, newdata = boot_test_expd,type="vector")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(pred_labels,real_labels, positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #MCCV
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- tree(labels ~ . - labels, data = mccv_train_expd, control = control_params)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="vector")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  
  
  #cv
  set.seed(123)
  folds <- createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    train_test_data <- train_expd[-train_idx, ]
    model <- tree(labels ~ . - labels, data = train_data, control = control_params)
    predictions_cv <- predict(model, newdata = train_test_data, type = "vector")[,2]
    roc_curve_cv <- pROC::roc(train_test_data$labels, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(train_test_data$labels == 1, "positive", "negative"))
    #real_labels_cv <- train_test_data$labels
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
  all_result_summary[[result_label]] <- all_result
  result_metrics
  importance_values <- numeric(length(used_features))
  names(importance_values) <- used_features
  non_leaf_nodes <- which(model$frame$var != "<leaf>")
  for (i in non_leaf_nodes) {
    feature <- as.character(model$frame$var[i])
    dev_reduction <- model$frame$dev[i] - sum(model$frame$dev[model$children[[i]]])
    importance_values[feature] <- importance_values[feature] + dev_reduction
  }
  importance_df <- data.frame(
    Overall = importance_values,
    row.names = names(importance_values)
  ) %>% 
    arrange(-Overall) %>% 
    filter(Overall > 0)
  all_result_importance[[paste0(feature_name, " & DT")]] <- importance_df
  print(importance_df)
}
##############################################################################
#######################################' 3.5 Random forest
##############################################################################
feature_sets <- list(
  DT = DT,
  #RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  test_expd <- as.matrix(test_exp)[, selected_features]
  testlabels <- factor(test_labels)
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  ctrl <- trainControl(method = "cv", number = 2,classProbs = TRUE,    
                       summaryFunction = twoClassSummary, 
                       verboseIter = TRUE)
  grid <- expand.grid(mtry = 7)
  set.seed(123)
  weights <- ifelse(train_expd$labels == "positive", 8, 1)
  model <- caret::train(labels ~ .,
                        data = train_expd,
                        method = "rf",   
                        trControl = ctrl,
                        tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
  pred.RF <- predict(model,newdata=train_expd,type="prob")
  pred.RF <- predict(model,newdata=train_expd,type="prob")
  roc_RFtrain <- pROC::roc(train_expd$labels, pred.RF[, 2])
  roc_RFtrain
  pred.RFtest <- predict(model,newdata=test_expd,type="prob")
  roc_RFtest <- pROC::roc(testlabels, pred.RFtest[, 2])
  roc_RFtest
  importance <- varImp(model)
  plot(importance)
  print(importance)
  importance <- varImp(model)  
  non_zero_importance <- as.data.frame(importance$importance)
  non_zero_importance <- non_zero_importance[non_zero_importance$Overall > 0, , drop = FALSE] 
  all_result_importance[[paste0(feature_name," & RF")]] <- non_zero_importance
  train_probs <- predict(model, type="prob")[,2]
  roc_curve <- roc(train_expd$labels, train_probs)
  roc_curve
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti<- optimal_coords$threshold
  #####
  train_result <- as.data.frame(predict(model, type="prob")[,2])
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "prob")[,2])
    tresult$type <- factor(ifelse(tresult[, 1] > cuti, "positive", "negative"), levels = levels_to_use)
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & RF")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BT
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  set.seed(123)
  n_bootstrap <- 1000 
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- boot_test_expd$labels
    weights <- ifelse(boot_train_expd$labels == "positive", 8, 1)
    model <- caret::train(labels ~ .,
                          data = boot_train_expd,
                          method = "rf",  
                          trControl = ctrl,
                          tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
    
    pred_probs <- predict(model, newdata = boot_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- boot_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7 
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    weights <- ifelse(mccv_train_expd$labels == "positive", 8, 1)
    model <- caret::train(labels ~ .,
                          data = mccv_train_expd,
                          method = "rf",  
                          trControl = ctrl,
                          tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  folds <- createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    train_labels_CV <- train_labels[train_idx]
    test_data <- train_expd[-train_idx, ]
    CV_labels <- train_labels[-train_idx]
    weights <- ifelse(train_data$labels == "positive", 8, 1)
    set.seed(123)
    model <- caret::train(labels ~ .,
                          data = train_data,
                          method = "rf",  
                          trControl = ctrl,
                          tuneGrid = grid, ntree= 300, sampsize = 350,nodesize = 30,weights = weights)
    
    predictions_cv <- predict(model, newdata =  test_data, type = "prob")[,2]
    roc_curve_cv <- pROC::roc(CV_labels, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(CV_labels == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ] 
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################' 3.6 XGBOOST
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  #XGBOOST = XGBOOST,
  GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
all_labels <- bind_rows(labels_list)
rownames(all_labels) <- all_labels[,1]
all_labels[,2] <- ifelse(all_labels[,2]==type, 1, 0)
train_exp <- exp_list[[which(exp_file==train_exp_names)]]
test_explist <- exp_list[which(!exp_file==train_exp_names)]
com <- intersect(rownames(train_exp), rownames(all_labels))
train_labels <- all_labels[com,]
train_exp <-train_exp[com,]
train_labels <- train_labels[, 2]

test_exp_names <- "DatasetB.txt"
test_exp <- exp_list[[which(exp_file==test_exp_names)]]
comtest <- intersect(rownames(test_exp), rownames(all_labels))
test_labels <- all_labels[comtest,]
test_exp <-test_exp[comtest,]
test_labels <- test_labels[, 2]
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  
  train_expd <- as.data.frame(train_exp[, selected_features])
  test_expd <- as.data.frame(test_exp[, selected_features])
  xgb_train <- xgb.DMatrix(data = data.matrix(train_expd), label = train_labels, weight = ifelse(train_labels == 1, 5, 1)) 
  xgb_test <- xgb.DMatrix(data = data.matrix(test_expd), label = test_labels) 
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1,  
    max_depth = 4, 
    subsample = 0.5, 
    colsample_bytree = 0.8,
    gamma = 0.1,
    min_child_weight = 4,lambda=1,alpha=1,scale_pos_weight = 9
  )
  watchlist <- list(train = xgb_train, test = xgb_test)
  set.seed(123) 
  model <- xgb.train(params = params, data = xgb_train, nrounds = 500, watchlist = watchlist, early_stopping_rounds = 50)
  importance_matrix <- xgb.importance(feature_names = colnames(xgb_train), model = model)
  print(importance_matrix)
  importance_single_column <- data.frame(Feature = importance_matrix$Feature, Importance = importance_matrix$Gain)
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & XGBOOST")]] <- as.data.frame(importance_filtered)
  pred.xgb <- predict(model,newdata=xgb_train,type="prob")
  roc_xgbtrain <- pROC::roc(train_labels, pred.xgb)
  roc_xgbtrain
  pred.xgbtest <- predict(model,newdata=xgb_test ,type="prob")
  roc_xgbtest <- pROC::roc(test_labels, pred.xgbtest)
  roc_xgbtest
  optimal_coords <- coords(roc_xgbtrain, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model,newdata=xgb_train,type="prob"))
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, newdata=xgb_test, type = "prob"))
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(test_labels == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & XGBOOST")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BT
  set.seed(123)
  n_bootstrap <- 1000 
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(xgb_train), size = nrow(xgb_train), replace = TRUE)
    boot_train_expd <- train_exp[sample_idx, selected_features]
    boot_labels <- train_labels[sample_idx]
    boot_test_expd <- train_exp[-sample_idx, selected_features]
    boot_test_labels <- train_labels[-sample_idx]
    btxgb_test <- xgb.DMatrix(data = boot_test_expd, label = boot_test_labels) 
    dtrain <- xgb.DMatrix(data = boot_train_expd , label = boot_labels)
    watchlist <- list(train = dtrain , test = btxgb_test)
    model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0,
                       watchlist = watchlist, early_stopping_rounds = 50)
    pred_probs <- predict(model, newdata = btxgb_test,type="prob")
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics<- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_exp[sample_idx, selected_features]
    mccv_train_labels <- train_labels[sample_idx]
    mccv_test_expd <- train_exp[-sample_idx, selected_features]
    mccv_test_labels <- train_labels[-sample_idx]
    mccvxgb_test <- xgb.DMatrix(data = mccv_test_expd, label = mccv_test_labels) 
    dtrain <- xgb.DMatrix(data = mccv_train_expd , label = mccv_train_labels)
    watchlist <- list(train = dtrain , test = mccvxgb_test)
    model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0,watchlist = watchlist
                       , early_stopping_rounds = 50)
    
    pred_probs <- predict(model, newdata = mccv_test_expd ,type="prob")
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(pred_labels,real_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  folds <- createFolds(train_labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_exp[train_idx, selected_features]
    train_labels_cv <- train_labels[train_idx]
    CV_data <- train_exp[-train_idx,selected_features ]
    CV_labels <- train_labels[-train_idx]
    dtrain <- xgb.DMatrix(data = train_data , label = train_labels_cv)
    dtest <- xgb.DMatrix(data = CV_data , label = CV_labels)
    watchlist <- list(train = dtrain , test = dtest)
    set.seed(123)
    model <- xgb.train(params = params, data = dtrain, nrounds = 500, watchlist = watchlist,
                       early_stopping_rounds = 50)
    predictions_cv <- predict(model, newdata = CV_data, type = "vector")
    roc_curve_cv <- pROC::roc(CV_labels, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(CV_labels == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################' 3.7Support vector machine
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  #SVM = SVM_RFE,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  test_expd <- as.data.frame(test_exp[, selected_features])
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  param_grid <- expand.grid(
    C = 0.02)
  set.seed(123)
  model <- caret::train(
    labels ~ .,                   
    data = train_expd,
    method = "svmLinear",
    tuneGrid = param_grid,
    trControl = fitControl)
  pred.svm <- predict(model,newdata=train_expd,type="prob")
  roc_curve <- pROC::roc(train_expd$labels, pred.svm[,2])
  roc_curve
  pred.svmtest <- predict(model,newdata=test_expd,type="prob")
  roc_curvetest <- pROC::roc(test_labels, pred.svmtest[, 2])
  roc_curvetest
  importance <- varImp(model)
  print(importance)
  plot(importance)
  importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & SVM")]] <- as.data.frame(importance_filtered)
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, type="prob"))
  train_result$type <- factor(ifelse(train_result[,2] > cuti, "positive", "negative") )
  train_result <- train_result[,-1]
  colnames(train_result) <- c("predict_p", "predict_result")
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, expdata, type = "prob"))
    tresult$type <- factor(ifelse(tresult[, 2] > cuti, "positive", "negative"), levels = levels_to_use)
    tresult <- tresult[,-1]
    colnames(tresult) <- c("predict_p", "predict_result")
    tresult$real_label <- factor(ifelse(labelsdata == 1, "positive", "negative"), levels = levels_to_use)
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & SVM")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative", "positive"))
    x$real_label <- factor(x$real_label, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p), levels = c("negative", "positive"), direction = "<"))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
    
    model <- caret::train(
      labels ~ .,                   
      data = boot_train_expd,
      method = "svmLinear",
      tuneGrid = param_grid,
      trControl = train_control)
    pred_probs <- predict(model, newdata = boot_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- boot_test_labels
    pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100  
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
    
    model <- caret::train(
      labels ~ .,                   
      data = mccv_train_expd,
      method = "svmLinear",
      tuneGrid = param_grid,
      trControl = train_control)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="prob")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = c("negative", "positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #cv
  set.seed(123)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  folds <- createFolds(train_labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    CV_labels <- train_labels[train_idx]
    test_data <- train_expd[-train_idx, ]
    test_labels_cv <- train_labels[-train_idx]
    train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
    model <- caret::train(
      labels ~ .,                  
      data = train_data,
      method = "svmLinear",
      tuneGrid = param_grid, metric = "ROC",
      trControl = train_control)
    predictions_cv <- predict(model, newdata = test_data, type = "prob")[,2]
    roc_curve_cv <- pROC::roc(test_labels_cv, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(test_labels_cv == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}

##############################################################################
#######################################' 3.8 Grandient Boosting Machine
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  #GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
all_labels <- bind_rows(labels_list)
rownames(all_labels) <- all_labels[,1]
all_labels[,2] <- ifelse(all_labels[,2]==type, 1, 0)
train_exp <- exp_list[[which(exp_file==train_exp_names)]]
test_explist <- exp_list[which(!exp_file==train_exp_names)]
com <- intersect(rownames(train_exp), rownames(all_labels))
train_labels <- all_labels[com,]
train_exp <-train_exp[com,]
train_labels <- train_labels[, 2]

test_exp_names <- "DatasetB.txt"
test_exp <- exp_list[[which(exp_file==test_exp_names)]]
comtest <- intersect(rownames(test_exp), rownames(all_labels))
test_labels <- all_labels[comtest,]
test_exp <-test_exp[comtest,]
test_labels <- test_labels[, 2]

test_expd <- as.data.frame(test_exp)
test_expd$labels <- test_labels
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  test_expd <- as.data.frame(test_exp[, selected_features])
  train_expd$labels <- as.numeric(train_expd$labels)
  set.seed(123)
  model <- gbm(
    labels ~ . - labels, 
    data = train_expd, 
    distribution = "bernoulli",   
    n.trees = 300,                
    interaction.depth = 5,         
    shrinkage = 0.01,              
    n.minobsinnode = 10,            
    bag.fraction = 0.5,            
    train.fraction = 1,         
    verbose = TRUE
  )
  all_result_importance[[paste0(feature_name," & GBM")]] <- as.data.frame(summary.gbm(model))[,-1,drop=F]
  pred.GBM <- predict(model,newdata=train_expd,type="response")
  roc_curve <- pROC::roc(train_expd$labels, pred.GBM)
  roc_curve
  pred.GBMtest <- predict(model, newdata = test_expd, type = "response")
  roc_GBMtest <- pROC::roc(test_labels, pred.GBMtest)
  roc_GBMtest
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  importance <- summary(model)
  print(importance)
  importance_data <- as.data.frame(summary(model, plotit = FALSE))
  non_zero_features <- importance_data[importance_data$rel.inf > 0, ]
  GBM <- rownames(non_zero_features)
  train_result <- as.data.frame(predict.gbm(model, as.data.frame(train_exp),type = "response"))  
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  colnames(train_result) <- c("predict_p", "predict_result","real_label")
  all_result <- lapply(test_explist, function(data, model, labelsdata){
    data = as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd,2]
    expdata <- data[comd,]
    tresult <- as.data.frame(predict(model, as.data.frame(expdata),type = "response"))
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
    colnames(tresult) <- c("predict_p", "predict_result","real_label")
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & GBM")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- gbm(
      labels ~ . - labels, 
      data = boot_train_expd, 
      distribution = "bernoulli",    
      n.trees = 300,             
      interaction.depth = 5,         
      shrinkage = 0.01,          
      n.minobsinnode = 10,          
      bag.fraction = 0.5,           
      train.fraction = 1,                
      verbose = TRUE             
    )
    
    pred_probs <- predict(model, newdata = boot_test_expd,type="response")
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_test_expd <- train_expd[-sample_idx, ]
    model <- gbm(
      labels ~ . - labels, 
      data = mccv_train_expd, 
      distribution = "bernoulli",   
      n.trees = 300,             
      interaction.depth = 5,       
      shrinkage = 0.01,           
      n.minobsinnode = 10,        
      bag.fraction = 0.5,          
      train.fraction = 1,                
      verbose = TRUE                 
    )
    pred_probs <- predict(model, newdata = mccv_test_expd,type="response")
    roc_curve <- pROC::roc(mccv_test_expd$labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_expd$labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  result_metrics 
  #CV
  set.seed(123)
  folds <- caret::createFolds(train_expd$labels, k = 10, returnTrain = TRUE)
  cv_results <- list()
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    train_data <- train_expd[train_idx, ]
    CV_labels <- train_labels[train_idx]
    test_datacv <- train_expd[-train_idx, ]
    test_labelscv <- train_labels[-train_idx]
    model <- gbm(
      labels ~ . - labels, 
      data = train_data, 
      distribution = "bernoulli", 
      n.trees = 300,           
      interaction.depth = 5,    
      shrinkage = 0.01,           
      n.minobsinnode = 10,           
      bag.fraction = 0.5,        
      train.fraction = 1,                  
      verbose = TRUE             
    )
    predictions_cv <- predict(model, newdata = test_datacv, type = "response")
    roc_curve_cv <- pROC::roc(test_labelscv, predictions_cv)
    optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
    cuti_cv <- optimal_coords_cv$threshold
    cv_results[[i]] <- list(model = model, roc_curve = roc_curve_cv, threshold = cuti_cv)
    pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
    real_labels_cv <- factor(ifelse(test_labelscv == 1, "positive", "negative"))
    pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
    conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
    accuracy_cv <- conf_matrix_cv$overall['Accuracy']
    recall_cv <- conf_matrix_cv$byClass['Recall']
    precision_cv <- conf_matrix_cv$byClass['Precision']
    f1_score_cv <- conf_matrix_cv$byClass['F1']
    npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
    ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve_cv)
    cv_results[[i]] <- c(
      accuracy = as.numeric(accuracy_cv), 
      recall = as.numeric(recall_cv), 
      precision = as.numeric(precision_cv), 
      f1_score = as.numeric(f1_score_cv), 
      npv = as.numeric(npv_cv), 
      ppv = as.numeric(ppv_cv), 
      auc = as.numeric(auc_value)
    )
  }
  cv_results_df <- do.call(rbind, cv_results)
  mean_accuracy <- mean(cv_results_df[, "accuracy"], na.rm = TRUE)
  mean_auc <- mean(cv_results_df[, "auc"], na.rm = TRUE)
  mean_recall <- mean(cv_results_df[, "recall"], na.rm = TRUE)
  mean_precision <- mean(cv_results_df[, "precision"], na.rm = TRUE)
  mean_f1 <- mean(cv_results_df[, "f1_score"], na.rm = TRUE)
  mean_npv <- mean(cv_results_df[, "npv"], na.rm = TRUE)
  mean_ppv <- mean(cv_results_df[, "ppv"], na.rm = TRUE)
  cv_results <- list()
  cv_metrics <- c(
    accuracy = mean_accuracy, recall = mean_recall, precision = mean_precision, 
    f1_score = mean_f1, npv = mean_npv, ppv = mean_ppv, auc = mean_auc 
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################'3.9 stepwise+Logistic Regression
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  #StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE,
  NB=NB_RFE
)
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  test_expd_df <- as.data.frame(test_exp[, selected_features])
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  fullModel = glm(labels~.-labels, family = 'binomial', data = train_expd) 
  nullModel = glm(labels ~ 1, family = 'binomial', data = train_expd) 
  model <- stepAIC(nullModel, 
                   direction = 'both',
                   scope = list(upper = fullModel, 
                                lower = nullModel),
                   trace = 0) 
  model_formula <- formula(model)
  StepWiseAICCV <- all.vars(model_formula)[-1]
  pred.step <- predict(model,newdata=train_expd,type="response")
  roc_curve <- pROC::roc(train_expd$labels, pred.step)
  roc_curve
  pred.steptest <- predict(model,newdata=test_expd_df,type="response")
  roc_curvetest <- pROC::roc(test_labels, pred.steptest)
  roc_curvetest
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, as.data.frame(train_exp),type = "response"))  
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  colnames(train_result) <- c("predict_p", "predict_result","real_label")
  all_result <- lapply(test_explist, function(data, model, labelsdata){
    data = as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd,2]
    expdata <- data[comd,]
    tresult <- as.data.frame(predict(model, as.data.frame(expdata),type = "response"))
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    tresult$real_label <- factor(ifelse(labelsdata==1, "positive", "negative"))
    colnames(tresult) <- c("predict_p", "predict_result","real_label")
    return(tresult)
  }, model=model, labelsdata=all_labels)
  
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & StepWise-AIC+LR")
  result_metrics <- sapply(all_result, function(x) {
    x$predict_result <- factor(x$predict_result, levels = c("negative","positive"))
    x$real_label <- factor(x$real_label, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(x$real_result,x$predict_label,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  result_metrics
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    boot_train_expd <- boot_train_expd[, c(StepWiseAICCV, "labels")]
    model <- glm(labels ~ ., family = "binomial",data = boot_train_expd)
    pred_probs <- predict(model, newdata = boot_test_expd,type="response")
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    levels(boot_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    mccv_train_expd <- mccv_train_expd[, c(StepWiseAICCV, "labels")]
    model <- glm(labels ~ ., family = "binomial",data = mccv_train_expd)
    pred_probs <- predict(model, newdata = mccv_test_expd,type="response")
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels,pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_data <- train_expd[, c(StepWiseAICCV, "labels")]
  model_cv <- caret::train(
    labels ~ .,
    data = model_data,
    method = "glm",                  
    family = "binomial",             
    metric = "ROC",                   
    trControl = train_control        
  )
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  importance <- varImp(model_cv)
  non_zero_importance <- as.data.frame(importance$importance)
  non_zero_importance <- non_zero_importance[non_zero_importance$Overall > 0, , drop = FALSE]  # 筛选重要性大于0的特征
  all_result_importance[[paste0(feature_name," & StepWise-AIC+LR")]] <- non_zero_importance
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = c("negative","positive"))
  conf_matrix_cv <- confusionMatrix(real_labels_cv,pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ]  # 存储AUC结果
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
#######################################'3.10 Naive Bayesian algorithm
##############################################################################
feature_sets <- list(
  DT = DT,
  RF = RF,
  XGBOOST = XGBOOST,
  GBM = GBM,
  StepWiseAIC = StepWiseAIC,
  SVM = SVM,
  KNN = KNN_RFE,
  LDA = LDA_RFE,
  RFE=RFE
  #NB=NB_RFE
)
levels_to_use <- c("negative", "positive")
for (feature_name in names(feature_sets)) {
  selected_features <- feature_sets[[feature_name]]
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  set.seed(123)
  model <- naiveBayes(labels ~ ., data = train_expd)
  train_probs <- predict(model,newdata=train_expd, type = "raw")
  roc_curve <- roc(train_labels, train_probs[,2])
  roc_curve
  optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
  cuti <- optimal_coords$threshold
  train_result <- as.data.frame(predict(model, as.data.frame(train_exp),type = "raw")[,2])  
  train_result$type <- factor(ifelse(train_result[,1] > cuti, "positive", "negative") )
  train_result$real_label <- factor(ifelse(train_labels==1, "positive", "negative"))
  colnames(train_result) <- c("predict_p", "predict_result","real_label")
  all_result <- lapply(test_explist, function(data, model, labelsdata) {
    data <- as.data.frame(data)
    comd <- intersect(rownames(data), rownames(labelsdata))
    labelsdata <- labelsdata[comd, 2]
    expdata <- data[comd, ]
    tresult <- as.data.frame(predict(model, as.data.frame(test_exp),type = "raw")[,2])  
    tresult$type <- factor(ifelse(tresult[,1] > cuti, "positive", "negative") )
    tresult$real_label <- factor(ifelse(test_labels==1, "positive", "negative"))
    colnames(tresult) <- c("predict_p", "predict_result","real_label")
    return(tresult)
  }, model = model, labelsdata = all_labels)
  all_result[[length(all_result) + 1]] <- train_result
  result_label <- paste0(feature_name," & NaiveBayes")
  result_metrics <- sapply(all_result, function(x) {
    
    x$predict_result <- factor(x$predict_result, levels = levels_to_use)
    x$real_label <- factor(x$real_label, levels = levels_to_use)
    
    conf_matrix <- confusionMatrix(x$real_label, x$predict_result,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    roc_curve <- roc(x$real_label, as.numeric(as.character(x$predict_p)))
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  #BT
  set.seed(123)
  n_bootstrap <- 1000  
  bootstrap_results <- lapply(1:n_bootstrap, function(i) {
    sample_idx <- sample(1:nrow(train_expd), size = nrow(train_expd), replace = TRUE)
    boot_train_expd <- train_expd[sample_idx, ]
    boot_labels <- train_expd$labels[sample_idx]
    boot_test_expd <- train_expd[-sample_idx, ]
    boot_test_labels <- train_expd$labels[-sample_idx]
    model <- naiveBayes(labels ~ ., data = boot_train_expd) 
    pred_probs <- predict(model, newdata = boot_test_expd,type="raw")[,2]
    roc_curve <- pROC::roc(boot_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- factor(ifelse(boot_test_labels==1, "positive", "negative"))
    pred_labels <- factor(pred_labels, levels = c("negative","positive"))
    conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  bootstrap_avg_metrics <- sapply(bootstrap_results, function(x) unlist(x)) %>% rowMeans(na.rm = TRUE)
  result_metrics <- cbind(bootstrap_avg_metrics, result_metrics)
  #mccv
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- factor(train_labels)
  set.seed(123)
  n_mccv <- 100 
  train_ratio <- 0.7  
  mccv_results <- lapply(1:n_mccv, function(i) {
    sample_idx <- createDataPartition(train_labels, p = train_ratio, list = FALSE)
    mccv_train_expd <- train_expd[sample_idx, ]
    mccv_train_labels <- mccv_train_expd$labels
    mccv_test_expd <- train_expd[-sample_idx, ]
    mccv_test_labels <- mccv_test_expd$labels
    model <- naiveBayes(labels ~ ., data = mccv_train_expd) 
    
    pred_probs <- predict(model, newdata = mccv_test_expd,type="raw")[,2]
    roc_curve <- pROC::roc(mccv_test_labels, pred_probs)
    optimal_coords <- coords(roc_curve, "best", best.method = "youden", ret = "all")
    cuti <- optimal_coords$threshold
    mccv_test_labels <- factor(ifelse(mccv_test_labels==1, "positive", "negative"))
    levels(mccv_test_labels)
    pred_labels <- ifelse(pred_probs > cuti, "positive", "negative")
    real_labels <- mccv_test_labels
    pred_labels <- factor(pred_labels, levels = levels_to_use)
    conf_matrix <- confusionMatrix(real_labels, pred_labels,positive = "positive")
    accuracy <- conf_matrix$overall['Accuracy']
    recall <- conf_matrix$byClass['Recall']
    precision <- conf_matrix$byClass['Precision']
    f1_score <- conf_matrix$byClass['F1']
    npv <- conf_matrix$byClass['Neg Pred Value']
    ppv <- conf_matrix$byClass['Pos Pred Value']
    auc_value <- auc(roc_curve)
    
    metrics <- c(accuracy = accuracy, recall = recall, precision = precision, f1_score = f1_score, npv = npv, ppv = ppv, auc = auc_value)
    return(metrics)
  })
  
  mccv_avg_metrics <- colMeans(do.call(rbind, mccv_results), na.rm = TRUE)
  result_metrics <- cbind(mccv_avg_metrics, result_metrics)
  #CV
  train_expd <- as.data.frame(train_exp[, selected_features])
  train_expd$labels <- train_labels
  train_expd$labels <- factor(ifelse(train_expd$labels==1, "positive", "negative"))
  levels(train_expd$labels)
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
  set.seed(123)
  model_cv <- caret::train(labels ~ .,                   
                           data = train_expd,  
                           method = "nb",          
                           trControl = train_control, 
                           metric = "ROC")           
  
  resample_results <- model_cv$resample
  mean(resample_results$ROC) 
  auc_value <- mean(resample_results$ROC)
  importance <- varImp(model_cv)
  importance_single_column <- data.frame(Feature = rownames(importance$importance), Importance = importance$importance[, "positive"])
  importance_filtered <- importance_single_column[importance_single_column$Importance != 0, ]
  rownames(importance_filtered) <- importance_filtered$Feature
  importance_filtered <- importance_filtered[ , -1, drop = FALSE]
  all_result_importance[[paste0(feature_name," & NaiveBayes")]] <- as.data.frame(importance_filtered)
  predictions_cv <- predict(model_cv, newdata = train_expd, type = "prob")[,2]
  roc_curve_cv <- pROC::roc(train_expd$labels, predictions_cv)
  optimal_coords_cv <- coords(roc_curve_cv, "best", best.method = "youden", ret = "all")
  cuti_cv <- optimal_coords_cv$threshold
  pred_labels_cv <- ifelse(predictions_cv > cuti_cv, "positive", "negative")
  real_labels_cv <- factor(ifelse(train_labels == 1, "positive", "negative"))
  pred_labels_cv <- factor(pred_labels_cv, levels = levels_to_use)
  conf_matrix_cv <- confusionMatrix(real_labels_cv, pred_labels_cv,positive = "positive")
  accuracy_cv <- conf_matrix_cv$overall['Accuracy']
  recall_cv <- conf_matrix_cv$byClass['Recall']
  precision_cv <- conf_matrix_cv$byClass['Precision']
  f1_score_cv <- conf_matrix_cv$byClass['F1']
  npv_cv <- conf_matrix_cv$byClass['Neg Pred Value']
  ppv_cv <- conf_matrix_cv$byClass['Pos Pred Value']
  cv_metrics <- c(
    accuracy = accuracy_cv, recall = recall_cv, precision = precision_cv, 
    f1_score = f1_score_cv, npv = npv_cv, ppv = ppv_cv, auc = auc_value
  )
  result_metrics <- cbind(cv_metrics, result_metrics)
  rownames(result_metrics) <- gsub("\\..*", "", rownames(result_metrics))
  result_metrics <- result_metrics[, -c(4, 5)]
  result_metrics
  all_result_acc[[result_label]] <- result_metrics['accuracy', ]
  all_result_recall[[result_label]] <- result_metrics['recall', ]
  all_result_FS[[result_label]] <- result_metrics['f1_score', ]
  all_result_npv[[result_label]] <- result_metrics['npv', ]
  all_result_ppv[[result_label]] <- result_metrics['ppv', ]
  all_result_pre[[result_label]] <- result_metrics['precision', ]
  all_result_auc[[result_label]] <- result_metrics['auc', ] 
  all_result_summary[[result_label]] <- all_result
  result_metrics
}
##############################################################################
##############################################################################
all_importance_rank_mean <- c()
all_model_number <- c()
for (genei in unique(im_result_df$Variable)) {
  all_model_number <- c(all_model_number, length(which(im_result_df$Variable==genei)))
  all_importance_rank_mean <- c(all_importance_rank_mean, mean((im_result_df[which(im_result_df$Variable==genei),])[,"rank"]))
}
all_rank <- data.frame(Fearute=unique(im_result_df$Variable), Rank=all_importance_rank_mean, model_number=all_model_number)
all_rank[order(all_rank$model_number, decreasing = T), ]
all_rank$Fearute <- reorder(all_rank$Fearute, all_rank$model_number, FUN = function(x) mean(x))
all_rank$combined_score <- all_rank$model_number*2 -all_rank$Rank
top_14_features <- all_rank[order(all_rank$combined_score,decreasing = TRUE), ][1:14, "Fearute"]
top_14_features <- as.character(top_14_features)
top_14_features
