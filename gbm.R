library(dplyr)
library(tidyr)
library(caret)
library(ggplot2)
library(randomForest)
library(cvTools)
library(tree)

setwd("D:/0. AI - TCS ion/Module 1. Business Analytics - R/")

hr_train = read.csv("13. Projects/4. Human Resources/hr_train.csv",stringsAsFactors = F)

hr_test = read.csv("13. Projects/4. Human Resources/hr_test.csv",stringsAsFactors = F)

hr_test$left=NA

hr_train$data="train"
hr_test$data="test"

hr_df = rbind(hr_train,hr_test)

glimpse(hr_df)

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for(cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
    
  }
  data[,var]=NULL
  return(data)
}

cat_cols = names(hr_df)[sapply(hr_df, function(x) is.character(x))][-3]

for(cat in cat_cols){
  hr_df=CreateDummies(hr_df,cat,100)
}

sapply(hr_df,function(x) sum(is.na(x)))

glimpse(hr_df)

#hr_df$left=as.factor(hr_df$left)

train_df=hr_df %>% 
  filter(data=="train") %>% 
  select(-data)

test_df=hr_df%>% 
  filter(data=="test") %>% 
  select(-data,-left)


set.seed(123)
s=sample(1:nrow(train_df),0.7*nrow(train_df))
train_set=train_df[s,]
test_set=train_df[-s,]

library(gbm)

param=list(interaction.depth=c(1:7),
           n.trees=c(50,100,200,500,700),
           shrinkage=c(.1,.01,.001),
           n.minobsinnode=c(1,2,5,10))

subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

num_trials=10
my_params=subset_paras(param,num_trials)

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}
# Note: A good value for num_trials is around 10-20% of total possible 
# combination. It doesnt have to be always 10

## ----this code will take too long to run--------------------------------------------------------------
myauc=0

for(i in 1:num_trials){
  print(paste('starting iteration :',i))
  
  params=my_params[i,]
  
  k=cvTuning(gbm,
             left ~ satisfaction_level + last_evaluation + number_project + 
             average_montly_hours + time_spend_company + Work_accident + 
             sales_hr + sales_technical + salary_medium + 
             salary_low,
             data =train_set,
             tuning =params,
             args=list(distribution="bernoulli"),
             folds = cvFolds(nrow(train_set), K=10, type ="random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="response",n.trees=params$n.trees)
  )
  score.this=k$cv[,2]
  
  if(score.this>myauc){
    print(params)
    
    myauc=score.this
    print(myauc)
    
    best_params=params
  }
  
  print('DONE')
  
}

## ----these values are from a previous run--------------------------------------------------------------
best_params=data.frame(interaction.depth=5,
                       n.trees=700,
                       shrinkage=0.01,
                       n.minobsinnode=10)


## ------------------------------------------------------------------------
hr.gbm.final=gbm(left ~ satisfaction_level + last_evaluation + number_project + 
                   average_montly_hours + time_spend_company + Work_accident + 
                   sales_hr + sales_technical + salary_medium + 
                   salary_low,
                 data=train_df,
                 n.trees = best_params$n.trees,
                 n.minobsinnode = best_params$n.minobsinnode,
                 shrinkage = best_params$shrinkage,
                 interaction.depth = best_params$interaction.depth,
                 distribution = "bernoulli")

## ----use these for prediciton and submission on test data--------------------------------------------------------------
test.score=predict(hr.gbm.final,newdata=test_df,type='response',
                    n.trees = best_params$n.trees)
write.csv(test.score,"13. Projects/4. Human Resources/Rushikesh_Shinde_P4_part2.csv",row.names=F)
