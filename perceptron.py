# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:18:37 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 假設 evaluate_classifier_performance 函數已經定義
from evaluate_classifier_performance import evaluate_classifier_performance

# 設置 pandas 顯示選項，確保所有列都能顯示
pd.set_option('display.max_columns', None)  # 不限制列數
pd.set_option('display.width', 1000)  # 設置顯示寬度，避免換行

# 定義 alpha 和 penalty 的範圍
alpha_list = np.linspace(0.00001, 1, 15)
penality_list = ['l1', 'l2', 'elasticnet']

# 定義 StratifiedKFold 來進行內部交叉驗證
skf_model = StratifiedKFold(n_splits=5, shuffle=True)

# 定義 10 折資料集的範圍
num_folds = 10
max_iterations = 10

# 用來儲存每個 fold 的表現指標
metrics_dict = {
    'train_accuracy': [],
    'test_accuracy': [],
    'train_precision': [],
    'test_precision': [],
    'train_recall': [],
    'test_recall': [],
    'train_class0_f1': [],
    'test_class0_f1': [],
    'train_class1_f1': [],
    'test_class1_f1': [],
    'train_f1': [],
    'test_f1': [],
    'train_weighted_f1': [],
    'test_weighted_f1': [],
    'train_auc': [],
    'test_auc': []
}

# 遍歷 10 個 fold
for fold in range(1, num_folds + 1):
    print(f"\n=== 正在處理 Fold {fold} ===")
    
    # 動態載入每個 fold 的訓練和測試資料
    train_file = f'Data/kfold_fold_{fold}_train.csv'
    test_file = f'Data/kfold_fold_{fold}_test.csv'
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # 移除不需要的欄位（如果有）
    if 'Unnamed: 0' in df_train.columns:
        del df_train['Unnamed: 0']
    if 'Unnamed: 0' in df_test.columns:
        del df_test['Unnamed: 0']

    # 分離特徵和標籤
    df_train_class = pd.DataFrame(df_train['y'])    
    df_train_features = df_train.loc[:, df_train.columns != 'y']
    df_test_class = pd.DataFrame(df_test['y'])
    df_test_features = df_test.loc[:, df_test.columns != 'y']
    
    ### Perceptron 分類器 - 選擇最佳參數
    for t in range(0, max_iterations):
        print(f"--- Fold {fold}, Iteration: {t}")
        AVG_ACC = np.zeros(shape=[len(alpha_list), len(penality_list)])
        STD_ACC = np.zeros(shape=[len(alpha_list), len(penality_list)])
        
        x_count = 0
        for alpha_value in alpha_list:
            y_count = 0
            for penality in penality_list:
                temp_accuracy_list = []
                for train_subset_index, cv_index in skf_model.split(df_train_features, df_train_class):
                    df_train_features_subset = df_train_features.iloc[train_subset_index]
                    df_train_class_subset = df_train_class.iloc[train_subset_index]
                    df_train_features_cv = df_train_features.iloc[cv_index]
                    df_train_class_cv = df_train_class.iloc[cv_index]
                    
                    perceptron_model = Perceptron(penalty=penality, alpha=alpha_value, class_weight='balanced')
                    perceptron_model.fit(df_train_features_subset, df_train_class_subset.values.ravel())
                    score_value = perceptron_model.score(df_train_features_cv, df_train_class_cv)
                    temp_accuracy_list.append(score_value)
                
                AVG_ACC[x_count, y_count] = np.mean(temp_accuracy_list)
                STD_ACC[x_count, y_count] = np.std(temp_accuracy_list)
                y_count += 1
            x_count += 1
        
        if t == 0:
            final_AVG_ACC = AVG_ACC
            final_STD_ACC = STD_ACC
        else:
            final_AVG_ACC = np.dstack([final_AVG_ACC, AVG_ACC])
            final_STD_ACC = np.dstack([final_STD_ACC, STD_ACC])
    
    # 計算平均準確度和選擇最佳參數
    final_accuracy_mean_list = np.mean(final_AVG_ACC, axis=2)
    max_ind = np.unravel_index(np.argmax(final_accuracy_mean_list, axis=None), final_accuracy_mean_list.shape)

    chosen_alpha = alpha_list[max_ind[0]]
    chosen_penalty = penality_list[max_ind[1]]
    print(f"Fold {fold} - 最佳 alpha: {chosen_alpha}")
    print(f"Fold {fold} - 最佳 Penalty: {chosen_penalty}")

    # 使用最佳參數訓練最終模型
    perceptron_model_final = Perceptron(penalty=chosen_penalty, alpha=chosen_alpha, class_weight='balanced')
    perceptron_model_final = CalibratedClassifierCV(estimator=perceptron_model_final, cv=10, method='isotonic')
    perceptron_model_final.fit(df_train_features, df_train_class.values.ravel())
    
    # 預測訓練和測試資料
    predicted_train = perceptron_model_final.predict(df_train_features)
    predicted_test = perceptron_model_final.predict(df_test_features)

    predicted_prob_train = perceptron_model_final.predict_proba(df_train_features)
    predicted_prob_test = perceptron_model_final.predict_proba(df_test_features)

    # 評估表現並儲存結果
    print(f"\nFold {fold} 表現評估：")
    evaluate_classifier_performance(df_train_class, predicted_train, predicted_prob_train, 
                                    df_test_class, predicted_test, predicted_prob_test, 'y')

    # 計算並儲存每個 fold 的指標
    # 準確度
    train_accuracy = perceptron_model_final.score(df_train_features, df_train_class)
    test_accuracy = perceptron_model_final.score(df_test_features, df_test_class)
    metrics_dict['train_accuracy'].append(train_accuracy)
    metrics_dict['test_accuracy'].append(test_accuracy)

    # 精確度（Precision）、召回率（Recall）、F1 分數（按類別和整體）
    train_precision = precision_score(df_train_class, predicted_train, average='macro')
    test_precision = precision_score(df_test_class, predicted_test, average='macro')
    train_recall = recall_score(df_train_class, predicted_train, average='macro')
    test_recall = recall_score(df_test_class, predicted_test, average='macro')

    train_f1_per_class = f1_score(df_train_class, predicted_train, average=None)
    test_f1_per_class = f1_score(df_test_class, predicted_test, average=None)
    train_f1 = f1_score(df_train_class, predicted_train, average='macro')
    test_f1 = f1_score(df_test_class, predicted_test, average='macro')
    train_weighted_f1 = f1_score(df_train_class, predicted_train, average='weighted')
    test_weighted_f1 = f1_score(df_test_class, predicted_test, average='weighted')

    metrics_dict['train_precision'].append(train_precision)
    metrics_dict['test_precision'].append(test_precision)
    metrics_dict['train_recall'].append(train_recall)
    metrics_dict['test_recall'].append(test_recall)
    metrics_dict['train_class0_f1'].append(train_f1_per_class[0])
    metrics_dict['test_class0_f1'].append(test_f1_per_class[0])
    metrics_dict['train_class1_f1'].append(train_f1_per_class[1])
    metrics_dict['test_class1_f1'].append(test_f1_per_class[1])
    metrics_dict['train_f1'].append(train_f1)
    metrics_dict['test_f1'].append(test_f1)
    metrics_dict['train_weighted_f1'].append(train_weighted_f1)
    metrics_dict['test_weighted_f1'].append(test_weighted_f1)

    # AUC
    train_auc = roc_auc_score(df_train_class, predicted_prob_train[:, 1])
    test_auc = roc_auc_score(df_test_class, predicted_prob_test[:, 1])
    metrics_dict['train_auc'].append(train_auc)
    metrics_dict['test_auc'].append(test_auc)

# 計算平均值和標準差，並生成最終表格
metrics_summary = {}
for metric, values in metrics_dict.items():
    mean_value = np.mean(values)
    std_value = np.std(values)
    metrics_summary[metric] = f"{mean_value:.4f} ± {std_value:.4f}"

# 將結果整理成表格
summary_df = pd.DataFrame({
    'Accuracy': [metrics_summary['train_accuracy'], metrics_summary['test_accuracy']],
    'Precision': [metrics_summary['train_precision'], metrics_summary['test_precision']],
    'Recall': [metrics_summary['train_recall'], metrics_summary['test_recall']],
    'Class 0 F1': [metrics_summary['train_class0_f1'], metrics_summary['test_class0_f1']],
    'Class 1 F1': [metrics_summary['train_class1_f1'], metrics_summary['test_class1_f1']],
    'F1 Score': [metrics_summary['train_f1'], metrics_summary['test_f1']],
    'Weighted F1 Score': [metrics_summary['train_weighted_f1'], metrics_summary['test_weighted_f1']],
    'AUC': [metrics_summary['train_auc'], metrics_summary['test_auc']]
}, index=['Train', 'Test'])

# 輸出最終結論
print("\n=== 10 折交叉驗證最終結論 ===")
print("Model Evaluation")
print(summary_df)

summary_df.to_csv('model_evaluation_summary.csv')