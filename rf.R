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

hr_df$left=as.factor(hr_df$left)

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

param=list(mtry=c(5,10,15,20,25,35),
           ntree=c(50,100,200,500,700),
           maxnodes=c(5,10,15,20,30,50,100),
           nodesize=c(1,2,5,10)
)

# Function for selecting random subset of params

subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}


mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}

num_trials=80
my_params=subset_paras(param,num_trials)
my_params

myauc=0

for(i in 1:num_trials){
  print(paste('starting iteration :',i))
  
  params=my_params[i,]
  
  k=cvTuning(randomForest,
             left~., 
             data =train_set,
             tuning =params,
             folds = cvFolds(nrow(train_set), K=10, type ="random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="prob")
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

best_params=data.frame(mtry=15,
                       ntree=50,
                       maxnodes=100,
                       nodesize=10)

hr.rf.final=randomForest(left ~ satisfaction_level + last_evaluation + number_project + 
                           average_montly_hours + time_spend_company + Work_accident + 
                           sales_hr + sales_technical + salary_medium + 
                           salary_low,
                            mtry=best_params$mtry,
                            ntree=best_params$ntree,
                            maxnodes=best_params$maxnodes,
                            nodesize=best_params$nodesize,
                            data=train_set
)

myControl <- trainControl(
  method = "cv", 
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = F, # IMPORTANT!
  verboseIter = TRUE
)


model <- train(
  left ~ satisfaction_level + last_evaluation + number_project + 
    average_montly_hours + time_spend_company + Work_accident + 
    sales_hr + sales_technical + salary_medium + 
    salary_low,
  tuneLength = 3,
  data = train_set, 
  method = "ranger",
  trControl = myControl
  )


library(pROC)

val.score=predict(hr.rf.final,newdata = test_set,type='prob')[,2]

auc(roc(test_set$left,val.score))

test.prob.score= predict(hr.rf.final,newdata = bank_test,type='prob')[,2]
write.csv(test.prob.score,"",row.names = F)











library(ggplot2)

hr_train %>% 
  select(Work_accident,left) %>% 
  filter(Work_accident==1) %>% 
  count(left==1)

hr_train %>% 
  select(sales,average_montly_hours) %>% 
  arrange(desc(sales)) %>% 
  summarise(median(average_montly_hours))


ggplot(hr_train,aes(average_montly_hours))+geom_histogram()



#228 0.0487 No low 0.33 0.18 3 accounting Yes 
