import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score # type: ignore
import pandas as pd
from sklearn.base import clone
import plot as p
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from plot import *


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def calculate_regression_metrics(y_test, y_pred):
    """
    Calculate regression performance metrics

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like, optional
        Predicted probabilities for positive class

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R^2': r2_score(y_test, y_pred),
        'RMSLE': mean_squared_log_error(y_test, y_pred)**0.5,
    }

    return metrics

def outliers(df, feature):
    """
    Обнаруживает и выводит выбросы в признаке DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Исходный DataFrame
    feature : str
        Название столбца для анализа
    method : str
    Returns :
        results -- словарь с информацией о выбросах
    """
    results = {}
    data = df[feature].dropna()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
    results = {
        'outliers': outliers_iqr.index.tolist(),
        'count': len(outliers_iqr),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'values': outliers_iqr.values.tolist()
    }
    return results

def evaluate_regression(y_test, y_pred, model_name="Model", enable_plot=True):
    """
    Evaluate regression performance with comprehensive metrics and visualizations

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str, optional
        Name of the model for display purposes
    enable_plot : bool, optional
        Whether to display plots and detailed reports

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    # Calculate all metrics
    metrics = calculate_regression_metrics(y_test, y_pred)

    if enable_plot:

        # Print detailed report
        print_regression_report(metrics, model_name)

    # Return metrics dictionary (excluding plot data for cleaner output)
    return {k: v for k, v in metrics.items()}


def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, seed=None):
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = evaluate_regression(
        y_test=y_test,
        y_pred=y_pred,
        model_name=model_name,
        enable_plot=False
    )

    return metrics


def train_evaluate_model_cv(model, model_name, X, y,
                            preprocessor=None, cv=5, seed=None):
    """
    Train and evaluate a model using cross-validation and optional preprocessing.

    Args:
        model: The model to train and evaluate
        model_name: Name of the model for reporting
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        preprocessor: Preprocessing pipeline (e.g., StandardScaler, OneHotEncoder)
        cv: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing evaluation metrics
    """
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Create or extend pipeline with preprocessor and model
    if isinstance(preprocessor, Pipeline):
        # If preprocessor is already a pipeline, append the model to it
        preprocessor.steps.append(('model', model))
        pipeline = preprocessor
    elif preprocessor is not None:
        # Create new pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        # No preprocessor, just use the model
        pipeline = model

    # Scoring metrics for cross-validation
    scoring = {
        'mae': 'neg_mean_absolute_error',  # для MAE (отрицательное, потому что cross_validate максимизирует метрики)
        'mse': 'neg_mean_squared_error',   # для MSE
        'r2': 'r2',                        # коэффициент детерминации
        'rmsle': 'neg_root_mean_squared_log_error'  # для RMSLE (отрицательное; sklearn >=1.0)
    }

    # Perform cross-validation on training data
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    metrics = {
        'MAE': -cv_results['test_mae'].mean(),    
        'MSE': -cv_results['test_mse'].mean(),
        'R^2': cv_results['test_r2'].mean(),
        'RMSLE': -cv_results['test_rmsle'].mean()
    }

    return metrics


def train_evaluate_models_cv(models: list, X, y, preprocessor=None, cv=5, seed=None):
    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)
        current_preprocessor = clone(preprocessor)

        # Store metrics
        all_metrics[model_name] = train_evaluate_model_cv(
            current_model, model_name, X, y, current_preprocessor, cv, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df


def train_evaluate_models(models: list, X_train, y_train, X_test, y_test, seed=None):
    """
    Train and evaluate multiple classification models, then display a heatmap of the metrics.

    Parameters:
    -----------
    models : list
        List of tuples containing (model_name, model_instance) where model_instance is a scikit-learn compatible classifier
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    preprocessor : Pipeline or Transformer, optional
        Preprocessing pipeline to apply to the data before training
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all evaluation metrics for all models
    """

    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)

        # Store metrics
        all_metrics[model_name] = train_evaluate_model(
            current_model, model_name, X_train, y_train, X_test, y_test, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df


def winsorize_outliers(df, column_name, lower_bound=None, upper_bound=None):
    df = df.copy()

    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound

    return df

def calculate_vif_sklearn(df, features):
    """
    Calculate VIF using sklearn only
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_values = []
    
    for i, feature in enumerate(features):
        # Целевая переменная - текущий признак
        y = df[feature]
        
        # Признаки - все остальные признаки
        X_features = [f for f in features if f != feature]
        X = df[X_features]
        
        # Обучаем линейную регрессию
        model = LinearRegression()
        model.fit(X, y)
        
        # Вычисляем R^2
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        # Вычисляем VIF
        vif = 1. / (1. - r_squared) if r_squared < 1 else float('inf')
        vif_values.append(vif)
    
    vif_data["VIF"] = vif_values
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    return vif_data

def quick_vif_check_sklearn(df, features):
    """
    Быстрая проверка VIF с цветовой индикацией
    """
    vif_df = calculate_vif_sklearn(df, features)
    
    print("VIF ANALYSIS (using sklearn)")
    print("=" * 50)
    print("VIF > 10: ❌ High multicollinearity")
    print("VIF 5-10: ⚠️ Moderate multicollinearity") 
    print("VIF < 5: ✅ Acceptable\n")
    
    for _, row in vif_df.iterrows():
        if row['VIF'] > 10:
            status = "❌ HIGH"
        elif row['VIF'] > 5:
            status = "⚠️ MODERATE" 
        else:
            status = "✅ OK"
        
        print(f"{status} | {row['Feature']:25} | VIF: {row['VIF']:6.2f}")
    
    return vif_df