
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV

import logging
import shap
import os
from tqdm import tqdm

from Preprocess import getStringValue, getCircumstance, getPlayType, getPlayResult

def XGBClassifierCore(X_train, X_test, y_train, y_test, best_params):
    clf = xgb.XGBClassifier(objective='binary:logistic', seed=11, **best_params)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, y_pred_proba, clf, accuracy

def runBayesianOptimization(X_train, y_train):
    param_space = {
        'n_estimators': (50, 200),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'subsample': (0.7, 1.0),
        'colsample_bytree': (0.7, 1.0)
    }
    
    clf = xgb.XGBClassifier(objective='binary:logistic', seed=41)
    
    bayes_search = BayesSearchCV(estimator=clf, search_spaces=param_space, scoring='accuracy', cv=5, n_jobs=-1, n_iter=30, random_state=42)
    bayes_search.fit(X_train, y_train)
    
    best_params = bayes_search.best_params_
    return best_params

def runShap(model, X_train, X_test, target_name, dirPath, fraStr):
    logging.info(f"Calculating SHAP values for target: {target_name}")
    logging.info(f"X_test columns: {X_test.columns}")
    logging.info(f"X_test indices: {X_test.index}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"Model: {model}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        logging.info(f"Multi-class model detected. SHAP values list length: {len(shap_values)}")
        num_classes = len(shap_values)
    else:
        logging.info(f"Single-class model detected or unexpected SHAP values shape. SHAP values shape: {shap_values.shape}")
        shap_values = [shap_values]
        num_classes = 1
    
    feature_names = X_test.columns.tolist()
    
    logging.info(f"Feature names: {feature_names}")
    logging.info(f"SHAP values shape after selection: {shap_values[0].shape}")

    for class_shap_values in shap_values:
        assert class_shap_values.shape[1] == len(feature_names), "Mismatch between feature names and SHAP values dimensions"

    dirPath = dirPath + 'SHAP/'
    os.makedirs(dirPath, exist_ok=True)

    feature_names = np.array(feature_names)

    for class_idx in range(num_classes):
        class_name = getStringValue(target_name, class_idx)
        file_name = f'[Shap]{target_name}_{class_name}_bar_{fraStr}.png'
        
        shap.summary_plot(shap_values[class_idx], X_test, plot_type="bar", show=False, feature_names=feature_names)
        plt.savefig(os.path.join(dirPath, file_name))
        plt.close()


def plotConfusionMatrix(y_test, y_pred, target_name, dirPath, fraStr):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(f'Confusion Matrix for {target_name}')
    
    dirPath = dirPath + 'ConfusionMatrix/'
    os.makedirs(dirPath, exist_ok=True)
    plt.savefig(dirPath + f'[ConfusionMatrix]{target_name}_{fraStr}.png')
    plt.close()


def plotNormalizedConfusionMatrix(y_test, y_pred, target_name, dirPath, fraStr):
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(f'Normalized Confusion Matrix for {target_name}')
    
    dirPath = dirPath + 'ConfusionMatrix/'
    os.makedirs(dirPath, exist_ok=True)
    plt.savefig(dirPath + f'[NormalizedConfusionMatrix]{target_name}_{fraStr}.png')
    plt.close()


def saveClassificationReport(y_test, y_pred, target_name, dirPath, fraStr):
    report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in np.unique(y_test)])
    
    dirPath = dirPath + 'Report/'
    os.makedirs(dirPath, exist_ok=True)
    with open(os.path.join(dirPath, f'[ClassificationReport]{target_name}_{fraStr}.txt'), 'w') as f:
        f.write(report)

def saveResult(results, target_name, dirPath, fraStr):
    os.makedirs(dirPath, exist_ok=True)
    file_path = os.path.join(dirPath, f'{target_name}_results_{fraStr}.txt')
    
    with open(file_path, 'w') as f:
        for key, value in results.items():
            f.write(f"Results for {key}:\n")
            f.write(f"  Cross-validation scores: {value['cross_val_scores']}\n")
            f.write(f"  Average accuracy: {value['avg_accuracy']:.4f}\n")
            f.write("  Best parameters:\n")
            for param, param_value in value['best_params'].items():
                f.write(f"    {param}: {param_value}\n")
            f.write(f"  Accuracies: {value['accuracies']}\n")
            f.write(f"  Best accuracy: {value['best_accuracy']:.4f}\n")
            f.write("\n")


def saveClassifications(X_test, y_test, y_pred, y_pred_proba, target_name, dirPath, fraStr):
    os.makedirs(dirPath, exist_ok=True)
    classification_df = pd.DataFrame({
        'X_test': X_test.index,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    classification_df.to_csv(os.path.join(dirPath, f'{target_name}_classification_{fraStr}.csv'), index=False)


def convertFractionIntoString(fraction):
    return f"{fraction:.2f}".replace('.', '').zfill(3) # with three digits


def runPlayTypeClassification(df, fraction, n_splits):
    dirPath = '../PlayTypeClassification/Classification/'
    os.makedirs(dirPath, exist_ok=True)
    logging.basicConfig(filename=dirPath + 'classification.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if fraction == 1.0:
        df_sampled = df
    else:
        analysisSampleSize = int(df.shape[0]*fraction)
        df_sampled = df.sample(n=analysisSampleSize, random_state=17)
    
    X = getCircumstance(df_sampled)
    targets = ['playType', 'huddle', 'formation']
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=97)
    
    for target in tqdm(targets, desc="Processing Targets"):
        logging.info(f"------- Processing {target}... -------")
        y = df_sampled[target]
        
        model = xgb.XGBClassifier(objective='multi:softprob', seed=43)
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
        
        avg_accuracy = scores.mean()
        logging.info(f"{target} Cross-validation scores: {scores}")
        logging.info(f"{target} Average cross-validation accuracy: {avg_accuracy}")
        
        accuracies = []
        best_params = None
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            best_params = runBayesianOptimization(X_train, y_train)
            
            y_pred, y_pred_proba, best_clf, accuracy = XGBClassifierCore(X_train, X_test, y_train, y_test, best_params)
            
            accuracies.append(accuracy)

            fraStr = convertFractionIntoString(fraction)
            
            # classification specific plots and reports
            runShap(best_clf, X_train, X_test, target, dirPath, fraStr)
            plotConfusionMatrix(y_test, y_pred, target, dirPath, fraStr)
            plotNormalizedConfusionMatrix(y_test, y_pred, target, dirPath, fraStr)
            saveClassificationReport(y_test, y_pred, target, dirPath, fraStr)
            
            if accuracy == max(accuracies):
                saveClassifications(X_test, y_test, y_pred, y_pred_proba, target, dirPath, fraStr)
        
        results = {}
        results[target] = {
            'cross_val_scores': scores,
            'avg_accuracy': avg_accuracy,
            'best_params': best_params,
            'accuracies': accuracies,
            'best_accuracy': max(accuracies),
        }
        logging.info(f"{target} Average Classification Accuracy: {avg_accuracy}")
        logging.info(f"Best Parameters for {target}: {best_params}")
    
        saveResult(results, target, dirPath, fraStr)


def loadResult(directory, target_name, fraction):
    def printResult(results):
        for key, result in results.items():
            print(f"Results for {key}:")
            print(f"  Cross-validation scores: {result['cross_val_scores']}")
            print(f"  Average accuracy: {result['avg_accuracy']:.4f}")
            print("  Best parameters:")
            for param, value in result['best_params'].items():
                print(f"    {param}: {value}")
            print(f"  Accuracies: {result['accuracies']}")
            print(f"  Best accuracy: {result['best_accuracy']:.4f}")
    
    fraStr = str(int(fraction * 100))  # Convert fraction to string
    file_path = os.path.join(directory, f'{target_name}_results_{fraStr}.txt')
    
    results = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        key = None
        for line in lines:
            line = line.strip()
            if line.startswith("Results for "):
                key = line.replace("Results for ", "").replace(":", "")
                results[key] = {}
            elif line.startswith("Cross-validation scores:"):
                # Convert the string of numbers to a list of floats
                scores_str = line.split(": ", 1)[1]
                results[key]['cross_val_scores'] = [float(x) for x in scores_str.strip('[]').split()]
            elif line.startswith("Average accuracy:"):
                results[key]['avg_accuracy'] = float(line.split(": ", 1)[1])
            elif line.startswith("Best parameters:"):
                results[key]['best_params'] = {}
            elif line.startswith("    "):
                param, param_value = line.split(": ", 1)
                results[key]['best_params'][param.strip()] = float(param_value.strip())
            elif line.startswith("Accuracies:"):
                accuracies_str = line.split(": ", 1)[1]
                results[key]['accuracies'] = [float(x) for x in accuracies_str.strip('[]').split(',')]
            elif line.startswith("Best accuracy:"):
                results[key]['best_accuracy'] = float(line.split(": ", 1)[1])
    
    printResult(results)
    return results

def loadClassification(directory, target_name, fraction):
    fraStr = convertFractionIntoString(fraction)
    file_path = os.path.join(directory, f'{target_name}_classification_{fraStr}.csv')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    classification_df = pd.read_csv(file_path)
    return classification_df