import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV


# Output directory
output_dir = "ML_output_results2"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# 读取数据
file_path = "ProteinPeptide_ML.csv"
data = pd.read_csv(file_path)

# 获取列名
column_names = data.columns.tolist()
Y_item = column_names[4]
X_items = column_names[5:]

X_name = data['name']
X_method = data['method']

# 归一化 X_items 相关列
X_data = data[X_items]
X_data_filled = X_data.fillna(X_data.mean())
scaler = MinMaxScaler()
X_data_normalized = pd.DataFrame(scaler.fit_transform(X_data_filled), columns=X_items)
data[X_items] = X_data_normalized


# 随机抽样 unique_X_name，划分为训练集和测试集
unique_X_name = np.unique(X_name.tolist())
unique_X_method = np.unique(X_method.tolist())
train_name, test_name = train_test_split(unique_X_name, test_size=0.2, random_state=42)

# 根据 train_name 和 test_name 划分数据
train_data = data[data['name'].isin(train_name)]
test_data = data[data['name'].isin(test_name)]

# 获取训练集和测试集的 X 和 y
X_train = train_data[X_items]
y_train = train_data[Y_item]

X_test = test_data[X_items]
y_test = test_data[Y_item]

# 转换为 Tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
#1. Define function to save evaluation results


def save_results(output_dir, model_name, correlation, mse, r2):
    result_file = os.path.join(output_dir, f'{model_name}_evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Pearson correlation coefficient: {correlation:.2f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"R² Score: {r2:.2f}\n")
    print(f"{model_name} evaluation results saved to {result_file}")

# 初始化 test_data 的副本，防止原始数据被修改
test_data_with_predictions = test_data.copy()

# 修改 evaluate_model 函数，将预测结果直接添加到 test_data_with_predictions 中
def evaluate_model_and_add_predictions(model, X_train, y_train, X_test, y_test, model_name, test_data):
    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation, _ = pearsonr(y_test, y_pred)

    # 输出评估结果
    print(f"{model_name} - Mean Squared Error (MSE): {mse:.2f}")
    print(f"{model_name} - R² Score: {r2:.2f}")

    # 保存评估结果
    save_results(output_dir, model_name, correlation, mse, r2)

    # 保存模型
    model_file_path = os.path.join(output_dir, f'{model_name}_model.pkl')
    joblib.dump(model, model_file_path)
    print(f"{model_name} model saved to {model_file_path}")

    # 将预测结果添加到 test_data 的新列中
    test_data[f'{model_name}_Predicted'] = y_pred

    return y_test, y_pred, correlation

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

param_grid_lr = {
    'fit_intercept': [True, False]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False]
}

param_grid_ridge = {
    'alpha': [0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
}

param_grid_lasso = {
    'alpha': [0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'max_iter': [1000, 5000, 10000],
    'selection': ['cyclic', 'random']
}

param_grid_elasticnet = {
    'alpha': [0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 0.9],  # balance between L1 and L2
    'fit_intercept': [True, False],
    'max_iter': [1000, 5000, 10000]
}

param_grid_knn = {
    'n_neighbors': [3, 5, 10, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 50]
}

param_grid_gbr = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0],
    'loss': ['ls', 'lad', 'huber', 'quantile']
}

param_grid_bagging = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.8, 1.0],
    'max_features': [0.5, 0.8, 1.0],
    'bootstrap': [True, False]
}



# models = {
#     "Random_Forest":  GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5),
#     "Linear_Regression": GridSearchCV(LinearRegression(), param_grid_lr, cv=5),
#     "Support_Vector_Regression": GridSearchCV(SVR(), param_grid_svr, cv=5),
#     "XGBoost": GridSearchCV(XGBRegressor(random_state=42), param_grid_xgb, cv=5),
#     "Decision_Tree": GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
# }

models = {
    "Random_Forest": GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5,n_jobs=-1),
    "Linear_Regression": GridSearchCV(LinearRegression(), param_grid_lr, cv=5,n_jobs=-1),
    "Support_Vector_Regression": GridSearchCV(SVR(), param_grid_svr, cv=5,n_jobs=-1),
    "XGBoost": GridSearchCV(XGBRegressor(random_state=42), param_grid_xgb, cv=5,n_jobs=-1),
    "Decision_Tree": GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5,n_jobs=-1),
    "Ridge_Regression": GridSearchCV(Ridge(), param_grid_ridge, cv=5,n_jobs=-1),
    "Lasso_Regression": GridSearchCV(Lasso(), param_grid_lasso, cv=5,n_jobs=-1),
    "ElasticNet": GridSearchCV(ElasticNet(), param_grid_elasticnet, cv=5,n_jobs=-1),
    "KNN_Regression": GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5,n_jobs=-1),
    "Gradient_Boosting": GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gbr, cv=5, n_jobs=-1),
    "Bagging_Regression": GridSearchCV(BaggingRegressor(random_state=42), param_grid_bagging, cv=5, n_jobs=-1)
}

# Create output directory if it doesn't exist
output_dir = 'ML_output_results2'
os.makedirs(output_dir, exist_ok=True)

# 依次评估每个模型，并将预测结果添加到 test_data_with_predictions
for model_name, model in models.items():
    y_test, y_pred, correlation = evaluate_model_and_add_predictions(
        model, X_train, y_train, X_test, y_test, model_name, test_data_with_predictions
    )

    # 可视化结果
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, label=f'{model_name} Predictions')
    
    # 添加完美预测的线（y = x）
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

    # 添加相关系数注释
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, color='black')

    # 设置标签和标题
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted Values - {model_name}')
    plt.legend()

    # 保存图像
    plot_file_path = os.path.join(output_dir, f'{model_name}_true_vs_predicted.png')
    plt.savefig(plot_file_path)
    plt.close()

# # 最后统一保存包含所有预测列的 test_data
# test_data_file_path = os.path.join(output_dir, 'test_data_with_all_predictions.csv')
# test_data_with_predictions.to_csv(test_data_file_path, index=False)
# print(f"All model predictions saved to {test_data_file_path}")
top_n = 3  # 设定 top_n 的值

# 初始化一个字典来存储每个模型的正确预测次数
correct_predictions = {model_name: 0 for model_name in models.keys()}
total_predictions = {model_name: 0 for model_name in models.keys()}

for model_name, model in models.items():
    # 打印当前模型名称
    print(f"Evaluating model: {model_name}")
    
    pred_col = f'{model_name}_Predicted'
    
    for name in test_name:
        # 过滤出当前 name 对应的 test_data 数据
        for method in unique_X_method:
            name_data = test_data_with_predictions[(test_data_with_predictions['name'] == name) & (test_data_with_predictions['method'] == method)]

            # 如果数据不足 top_n，跳过
            if len(name_data) < top_n:
                print(f"Skipping {name} as it has less than {top_n} samples")
                continue

            # 提取预测值 top_n 对应的索引
            top_n_pred_indices = name_data.nlargest(top_n, pred_col).index
  
            # Get the boolean arrays for each condition
            condition_Y_item = name_data.loc[top_n_pred_indices, Y_item] > 0.8
            condition_pred_col = name_data.loc[top_n_pred_indices, pred_col] > 0.8

            # Calculate the number of matching True values
            matching_true_count = (condition_Y_item == condition_pred_col).sum()
            
            
            correct_predictions[model_name] += matching_true_count
            total_predictions[model_name] += top_n 

            # print(matching_true_count)
           

            # # 提取真实值 top_n 对应的索引
            # top_n_true_indices = name_data.nlargest(top_n, Y_item).index

            # # 比较 top_n 的索引是否完全一致
            # is_identical = set(top_n_pred_indices) == set(top_n_true_indices)
            
            # # 统计正确预测的数量
            # if is_identical:
            #     correct_predictions[model_name] += 1
            # total_predictions[model_name] += 1

# 计算每个模型的正确率
accuracy = {model_name: correct_predictions[model_name] / total_predictions[model_name] for model_name in models.keys()}

# 打印每个模型的正确率
for model_name, acc in accuracy.items():
    print(f"Accuracy of {model_name}: {acc * 100:.2f}%")