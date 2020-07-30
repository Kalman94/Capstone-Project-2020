############################################################

             ### MACHINE LEARNING PART ###

             ### MACHINE VISION DATASET ### 

##############################################################
import pandas as pd
from pycaret.classification import *
pd.set_option('display.max_columns', None)


dataset=pd.read_csv( "C:\\Users\\User\\Desktop\\Capstone\\1.Code\\Datasets\\MachineVision.csv", sep=",")
del dataset['Unnamed: 0']

#check the shape of data
dataset.shape

#Randomly choose and hide 5% -aka 14 images- for Predictions in the end
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
data.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##################
# Preprocessing  #
##################

clss_pre = setup(data = data, target ='status',
                 session_id=312, silent=True,
                  normalize = True,
                   numeric_features=['2','9','16','17','18','21','22','24','38','39','40','41',
                                     '42','43','44','45','46','47','48','49','50','51','52','53',
                                     '54','55','56','57','58','59','60','61','102','103','104',
                                     '105','106','107','108','109','117','118','119','120','121',
                                     '122','123','124','125','126','127','128','129','144','145',
                                     '146','147','148','149','150','151','152','154','155','160',
                                     '161','162','163','164','165','166','167','168','169','170',
                                     '173','174','175','176','177','178','179','180','181','182',
                                     '183','184','185','186','187','188','189','190'],
                  feature_selection=True,
                  feature_interaction=True,
                  remove_multicollinearity = True,
                  multicollinearity_threshold = 0.99)

#Test without parameter optimizasion

init_m = compare_models(sort='Precision',turbo=False)
init_m = init_m.data

#To CSV
#init_m.to_csv(r'C:\\Users\\User\\Desktop\\Capstone\\1.Code\\Datasets\\Produced_Datasets\\MachineVision_Init_Clf.csv')

#To Excel
from pandas import ExcelWriter

writer = ExcelWriter('C:\\Users\\User\\Desktop\\Capstone\\1.Code\\Datasets\\Produced_Datasets\\MachineVision_Init_Clf.xlsx')
#writer = ExcelWriter('C:\\Users\\User\\Desktop\\Capstone\\1.Code\\Datasets\\Produced_Datasets\\MachineVision(no_multi)_Init_Clf.xlsx')
init_m.to_excel(writer,'Without_Tunning')
writer.save()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################
# Hyperparameter Tuning #
#########################

#Using 10 fold stratified cross validation
#Create the models
dt = create_model('dt')
rf = create_model('rf')
lr = create_model('lr')
knn= create_model('knn')
ct = create_model('catboost')
lgbm = create_model('lightgbm')
svm= create_model('svm')
rfbsvm = create_model('rbfsvm')
nb = create_model('nb')
ada = create_model('ada')
nn= create_model('mlp')  
et = create_model('et')
ridge = create_model('ridge')
egb= create_model('xgboost')
gbc= create_model('gbc')

#GridSearchCV
#all hyperparameter tuning is set to optimize for Precision:
print("****** Decision Tree ******")
tuned_dt = tune_model('dt',optimize= 'Precision')
print('------------------------------------------------------\n')
print("****** Random Forest ******")
tuned_rf = tune_model('rf',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Logistic Regrassion ******")
tuned_lr = tune_model('lr',optimize='Precision')
print('------------------------------------------------------\n')
print("****** K Nearest  Neighbors******")
tuned_knn = tune_model('knn',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Cat Boost ******")
tuned_ct = tune_model('catboost',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Light Gradient Boost ******")
tuned_lgbm = tune_model('lightgbm',optimize='Precision')
print('------------------------------------------------------\n')
print("****** SVM ******")
tuned_svm = tune_model('svm',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Radial SVM ******")
tuned_rfbsvm = tune_model('rbfsvm',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Naive Bayes ******")
tuned_nb = tune_model('nb',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Adaptive Boosting ******")
tuned_ada = tune_model('ada',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Multiple Perceptron Neural Networks ******")
tuned_nn = tune_model('mlp',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Extra Trees  ******")
tuned_et = tune_model('et',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Ridge ******")
tuned_ridge = tune_model('ridge',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Optimized Distributed Gradient Boosting ******")
tuned_egb = tune_model('xgboost',optimize='Precision')
print('------------------------------------------------------\n')
print("****** Gradient Boosting Classifier ******")
tuned_gbc = tune_model('gbc',optimize='Precision')
print('------------------------------------------------------\n')

########################
# Optimized Parameters #
########################

# #Print optimized parameters for each:
# evaluate_model(tuned_dt)
# evaluate_model(tuned_rf)
# evaluate_model(tuned_lr)
# evaluate_model(tuned_knn)
# #evaluate_model(tuned_ct)
# evaluate_model(tuned_lgbm)
# evaluate_model(tuned_svm)
# evaluate_model(tuned_rfbsvm)
# evaluate_model(tuned_nb)
# evaluate_model(tuned_ada)
# evaluate_model(tuned_nn)
# evaluate_model(tuned_et)
# evaluate_model(tuned_ridge)
# evaluate_model(tuned_egb)
# evaluate_model(tuned_gbc)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################
#######  Plots  #########
#########################

#Random Forest
plot_model(tuned_rf, plot = 'auc')
plot_model(tuned_rf, plot = 'pr')
plot_model(tuned_rf, plot='feature')
plot_model(tuned_rf, plot = 'confusion_matrix')
plot_model(tuned_rf, plot = 'error')
plot_model(tuned_rf, plot = 'class_report')
plot_model(tuned_rf, plot='calibration')
interpret_model(tuned_rf,plot='summary')


#Cat Boost
interpret_model(tuned_ct,plot='summary')


#Ada Boost
plot_model(tuned_ada, plot = 'auc')
plot_model(tuned_ada, plot = 'pr')
plot_model(tuned_ada, plot='feature')
plot_model(tuned_ada, plot = 'confusion_matrix')
plot_model(tuned_ada, plot = 'error')
plot_model(tuned_ada, plot = 'class_report')
plot_model(tuned_ada, plot='calibration')

#Extra Trees
#plot_model(tuned_et, plot = 'rfe')
plot_model(tuned_et, plot = 'auc')
plot_model(tuned_et, plot = 'pr')
plot_model(tuned_et, plot='feature')
plot_model(tuned_et, plot = 'confusion_matrix')
plot_model(tuned_et, plot = 'error')
plot_model(tuned_et, plot = 'class_report')
plot_model(tuned_et, plot='calibration')
interpret_model(tuned_et,plot='summary')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#####################
# Ensemble learning #
#####################


## Stacking Models##

#Stacking is an ensemble learning technique that combines multiple models via a meta-model. 
#Another way to think about stacking is that multiple models are trained to predict the outcome 
#and a meta-model is created that uses the predictions from those models 
#as an input along with the original features. 
#The implementation of stack_models() is based on Wolpert, D. H. (1992b). Stacked generalization (Read More).

stacked = stack_models([tuned_rf, tuned_ct, tuned_et], meta_model=tuned_et,plot=True)

## Ensemble model & ##
## Probability Calibration ##

#Well calibrated classifiers are probabilistic and provide outputs 
#in the form of probabilities that can be directly interpreted as a confidence level.

#create a bagging model
et_boosted = create_model('et', ensemble = True, method = 'Bagging')
#calibrate trainde boosted dt
calibrated_et = calibrate_model(et_boosted)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################
#####  Predictios #######
#########################

#with extra tree
pred_et = predict_model(tuned_et ,data=data_unseen)

#with random forest
pred_rf=predict_model(tuned_rf,data=data_unseen)

#With stacked models
pred_stacked = predict_model(stacked ,data=data_unseen)

#With calibrating model
pred_cal = predict_model(calibrated_et ,data=data_unseen)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################
#### Save Models ########
#########################

save_model(tuned_et, 't_et_mv_2706')
save_model(tuned_rf, 't_rf_mv_2706')
save_model(stacked, 't_stacked_mv_2706')
save_model(calibrated_et, 't_cal_et_mv_2706')


save_experiment('mv_2706')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~