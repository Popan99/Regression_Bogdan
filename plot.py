import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
from matplotlib.colors import LinearSegmentedColormap
import phik


def plot_phik(data, figsize=(12, 8)):
    phik_matrix = data.phik_matrix()
    plt.figure(figsize=(10, 8))
    sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.show()

def plot_hist(df, feature):
    plt.figure(figsize = (6,6))
    sns.histplot(df[feature], stat = 'probability', kde = True)
    plt.title(f'Плотность распределения {feature}')

def plot_hist_numeric(data, feature, figsize=(8, 4), x_min=None, x_max=None):
    filtered_data = data.copy()
    if x_min is not None:
        filtered_data = filtered_data[filtered_data[feature] >= x_min]
    if x_max is not None:
        filtered_data = filtered_data[filtered_data[feature] <= x_max]

    plt.figure(figsize=figsize)
    plt.grid()
    sns.histplot(filtered_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


def plot_hist_categorical(data, feature, figsize=(4, 4)):
    category_counts = data[feature].value_counts()
    category_counts = category_counts.sort_values(ascending=False)
    plt.figure(figsize=figsize)
    plt.grid()
    sns.barplot(x=category_counts.values,
                y=category_counts.index,
                hue=category_counts.index,  # Add this
                palette="viridis",
                orient='h',
                legend=False)  # Add this
    plt.title(f'Distribution of {feature}')
    plt.ylabel(feature)
    plt.xlabel('Frequency')
    plt.show()


def plot_categorical_relationship(df, col1, col2):
    # Абсолютные значения
    count_crosstab = pd.crosstab(df[col1], df[col2])

    # Доли по строкам (внутри col1)
    row_prop = pd.crosstab(df[col1], df[col2], normalize='index')

    # Доли по столбцам (внутри col2)
    col_prop = pd.crosstab(df[col1], df[col2], normalize='columns')

    # Фигура с 3 подграфиками по горизонтали
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Абсолютные значения
    sns.heatmap(count_crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f'Абсолютные значения\n{col1} vs {col2}')
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel(col1)

    # 2. Доли внутри col1 (по строкам)
    sns.heatmap(row_prop, annot=True, fmt=".2f", cmap="Greens", ax=axes[1])
    axes[1].set_title(f'Доли внутри {col1} (по строкам)')
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel(col1)

    # 3. Доли внутри col2 (по столбцам)
    sns.heatmap(col_prop, annot=True, fmt=".2f", cmap="Oranges", ax=axes[2])
    axes[2].set_title(f'Доли внутри {col2} (по столбцам)')
    axes[2].set_xlabel(col2)
    axes[2].set_ylabel(col1)

    plt.tight_layout()
    plt.show()


def plot_numeric_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str = None,
    target_colors: dict = None,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None
):
    """
    Строит scatter plot зависимости между двумя числовыми переменными.
    При наличии бинарной таргетной переменной — точки окрашиваются по её значению.
    Позволяет задать ограничения на оси X и Y.

    :param df: pandas DataFrame
    :param x_col: Название числовой переменной по оси X
    :param y_col: Название числовой переменной по оси Y
    :param target_col: (опционально) Название бинарной переменной для окраски точек
    :param target_colors: (опционально) Словарь вида {значение_таргета: цвет}
    :param x_min: (опционально) Минимальное значение оси X
    :param x_max: (опционально) Максимальное значение оси X
    :param y_min: (опционально) Минимальное значение оси Y
    :param y_max: (опционально) Максимальное значение оси Y
    """
    # Проверка колонок
    for col in [x_col, y_col, target_col] if target_col else [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' отсутствует в DataFrame.")

    # Проверка типов
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise TypeError(f"{x_col} не является числовой переменной.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"{y_col} не является числовой переменной.")

    # Проверка бинарного таргета
    if target_col is not None:
        unique_vals = sorted(df[target_col].dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(
                f"Таргет '{target_col}' должен быть бинарным (2 уникальных значения).")

        # Палитра
        if target_colors is None:
            palette = {unique_vals[0]: 'blue', unique_vals[1]: 'red'}
        else:
            if not all(val in target_colors for val in unique_vals):
                raise ValueError(
                    f"target_colors должен содержать оба значения таргета: {unique_vals}")
            palette = target_colors

    # Построение графика
    plt.figure(figsize=(8, 6))
    if target_col:
        sns.scatterplot(data=df, x=x_col, y=y_col,
                        hue=target_col, palette=palette)
        plt.legend(title=target_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, color='blue')

    # Ограничения осей
    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    plt.title(f'Зависимость {y_col} от {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def print_regression_report(metrics, model_name="Model"):
    """
    Print regression evaluation report

    Parameters:
    -----------
    metrics : dict
        Dictionary containing all metrics (output from calculate_regression_metrics)
    model_name : str, optional
        Name of the model for display purposes
    """
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'R^2', 'RMSLE'],
        'Value': [
            f'{metrics["MAE"]:.4f}',
            f'{metrics["MSE"]:.4f}',
            f'{metrics["R^2"]:.4f}',
            f'{metrics["RMSLE"]:.4f}'
        ]
    })

    # Display results
    print("\n" + "="*60)
    print(f"{model_name.upper()} EVALUATION".center(60))
    print("="*60)

    print("\nMAIN METRICS:")
    print(metrics_df.to_string(index=False))


    print("\n" + "="*60)


def plot_feature_importance(model, feature_names, top_n=None, figsize=(10, 6),
                            model_type='auto'):
    """
    Plot feature importance for various model types using Seaborn.

    Parameters:
    - model: Trained model (DecisionTree, RandomForest, LogisticRegression, etc.)
    - feature_names: List of feature names
    - top_n: Show only top N important features (None for all)
    - figsize: Figure size
    - model_type: 'auto' (default), 'tree', or 'linear'. If 'auto', tries to determine automatically
    """
    # Determine model type if auto
    if model_type == 'auto':
        if hasattr(model, 'feature_importances_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            raise ValueError(
                "Could not determine model type automatically. Please specify 'tree' or 'linear'")

    # Get feature importances based on model type
    if model_type == 'tree':
        importances = model.feature_importances_
        importance_label = "Feature Importance"
    elif model_type == 'linear':
        importances = np.abs(model.coef_)
        importance_label = "Absolute Coefficient"
    else:
        raise ValueError("model_type must be either 'tree' or 'linear'")

    # Create DataFrame
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Select top_n features if specified
    if top_n is not None:
        feature_imp = feature_imp.head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature',
                data=feature_imp, palette='viridis')
    plt.title(f'Feature Importances ({model_type} model)')
    plt.xlabel(importance_label)
    plt.tight_layout()
    plt.show()

    return feature_imp


def visualize_decision_tree(model, feature_names, class_names=None,
                            figsize=(20, 10), max_depth=None):
    """
    Visualize the decision tree structure.

    Parameters:
    - model: Trained DecisionTree model
    - feature_names: List of feature names
    - class_names: List of class names (for classification)
    - figsize: Figure size
    - max_depth: Maximum depth to display (None for full tree)
    """
    plt.figure(figsize=figsize)
    plot_tree(model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              proportion=True,
              max_depth=max_depth)
    plt.title('Decision Tree Visualization')
    plt.show()


def plot_hyperparam_search_results(
    results,
    score_key='mean_test_score',
    title='Hyperparameter Tuning Results',
    xtick_step=5
):
    """
    Generic plot function for hyperparameter search results from GridSearchCV, RandomizedSearchCV,
    BayesSearchCV, or any source with similar output.

    Args:
        results (dict or pd.DataFrame): Search results. Must contain 'params' and score_key.
        score_key (str): Key for the score column (default 'mean_test_score').
        title (str): Plot title.
        xtick_step (int): Frequency of x-axis labels.
    """
    # Normalize input
    if isinstance(results, dict):
        params = results.get('params')
        scores = results.get(score_key)
        if params is None or scores is None:
            raise ValueError(
                f"'params' and '{score_key}' must exist in results dict.")
        df = pd.DataFrame(params)
        df[score_key] = scores
    elif isinstance(results, pd.DataFrame):
        if 'params' in results.columns:
            df = pd.DataFrame(results['params'].tolist())
            df[score_key] = results[score_key].values
        else:
            raise ValueError("DataFrame input must have a 'params' column.")
    else:
        raise TypeError(
            "results must be a dict (like cv_results_) or a DataFrame.")

    df = df.reset_index().rename(columns={'index': 'Set #'})

    # Best score
    best_idx = df[score_key].idxmax()
    best_score = df.loc[best_idx, score_key]

    # Plot
    plt.figure(figsize=(12, 6))
    x = df['Set #']
    y = df[score_key]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Hyperparameter Set #")
    plt.ylabel(score_key)
    plt.grid(True)

    # Clean x-ticks
    plt.xticks(ticks=x[::xtick_step])

    # Highlight best
    plt.plot(df.loc[best_idx, 'Set #'], best_score,
             'ro', label=f'Best: {best_score:.4f}')
    plt.annotate(f'Best\n{best_score:.4f}',
                 xy=(df.loc[best_idx, 'Set #'], best_score),
                 xytext=(df.loc[best_idx, 'Set #'], best_score + 0.02),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 ha='center')

    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


def compare_metrics_heatmap(df1, df2, df1_name='DF1', df2_name='DF2',
                            figsize=(8, 4), annot_fontsize=10,
                            title='Comparison of ML Metrics'):
    """
    Compare two DataFrames of ML metrics and plot a heatmap of their differences.

    Parameters:
    - df1, df2: DataFrames containing metrics for ML algorithms (algorithms as index, metrics as columns)
    - df1_name, df2_name: Names to display for each DataFrame in the comparison
    - figsize: Size of the output figure
    - annot_fontsize: Font size for annotations in heatmap
    - title: Title for the plot

    Returns:
    - A matplotlib Figure object
    - The delta DataFrame showing the differences
    """

    # Calculate delta (difference) between DataFrames
    delta = df2 - df1

    # Create a custom red-white-green colormap
    colors = ["#ff2700", "#ffffff", "#00b975"]  # Red -> White -> Green
    cmap = LinearSegmentedColormap.from_list("rwg", colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        delta,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=0,
        linewidths=.5,
        ax=ax,
        annot_kws={"size": annot_fontsize},
        cbar_kws={'label': f'Difference ({df2_name} - {df1_name})'}
    )

    # Customize plot
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    return fig, delta
