"""
            -- viz/plots.py --

Visualization functions for HPO results.

All functions accept a 'history' list of dicts with format:
    [{'params': {...}, 'score': float, 'trial': int}, ...]

Usage:
    from tinyhpo.viz import plots

    # After optimization
    plots.convergence(history)
    plots.parallel_coordinates(history)
    plots.marginal_plots(history)
    plots.heatmap(history, "learning_rate", "hidden_size")
    plots.param_importance(history)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed. Visualization functions will not work.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. "
                         "Install with: pip install matplotlib")


def _extract_data(history: List[Dict]) -> Tuple[List[Dict], List[float], List[str]]:
    """Extract params, scores, and param names from history."""
    params_list = [h['params'] for h in history]
    scores = [h['score'] for h in history]
    param_names = list(params_list[0].keys()) if params_list else []
    return params_list, scores, param_names


def convergence(history: List[Dict],
                title: str = "Convergence Plot",
                save_path: Optional[str] = None,
                show: bool = True) -> Optional[plt.Figure]:
    """
    Plot best score vs iteration number.

    Shows how the best-found score improves over the optimization process.

    Args:
        history: List of trial results
        title: Plot title
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        matplotlib Figure if not showing, else None
    """
    _check_matplotlib()

    params_list, scores, _ = _extract_data(history)

    # Calculate running best
    best_scores = []
    current_best = float('-inf')
    for score in scores:
        current_best = max(current_best, score)
        best_scores.append(current_best)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(1, len(scores) + 1)

    # Plot individual scores
    ax.scatter(iterations, scores, alpha=0.5, label='Trial scores', color='blue')

    # Plot best so far
    ax.plot(iterations, best_scores, 'r-', linewidth=2, label='Best so far')

    ax.set_xlabel('Trial')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        return None

    return fig


def parallel_coordinates(history: List[Dict],
                        title: str = "Parallel Coordinates",
                        colormap: str = "viridis",
                        save_path: Optional[str] = None,
                        show: bool = True) -> Optional[plt.Figure]:
    """
    Plot parallel coordinates showing all parameters and scores.

    Each line represents a trial, with color indicating the score.

    Args:
        history: List of trial results
        title: Plot title
        colormap: Matplotlib colormap name
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        matplotlib Figure if not showing, else None
    """
    _check_matplotlib()

    params_list, scores, param_names = _extract_data(history)

    # Normalize each parameter to [0, 1] for plotting
    normalized_data = []
    param_ranges = {}

    for param in param_names:
        values = [p[param] for p in params_list]
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            normalized = [0.5] * len(values)
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        normalized_data.append(normalized)
        param_ranges[param] = (min_val, max_val)

    # Add scores as last column
    score_min, score_max = min(scores), max(scores)
    if score_max == score_min:
        norm_scores = [0.5] * len(scores)
    else:
        norm_scores = [(s - score_min) / (score_max - score_min) for s in scores]
    normalized_data.append(norm_scores)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # X positions for each axis
    x_positions = list(range(len(param_names) + 1))

    # Create colormap
    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))

    # Plot each trial as a line
    for i in range(len(params_list)):
        y_values = [normalized_data[j][i] for j in range(len(param_names) + 1)]
        color = cmap(norm(scores[i]))
        ax.plot(x_positions, y_values, color=color, alpha=0.7, linewidth=1.5)

    # Set axis labels
    labels = param_names + ['score']
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add range annotations
    for i, param in enumerate(param_names):
        min_val, max_val = param_ranges[param]
        ax.annotate(f'{min_val}', (i, -0.05), ha='center', fontsize=8)
        ax.annotate(f'{max_val}', (i, 1.05), ha='center', fontsize=8)

    # Score range
    ax.annotate(f'{score_min:.3f}', (len(param_names), -0.05), ha='center', fontsize=8)
    ax.annotate(f'{score_max:.3f}', (len(param_names), 1.05), ha='center', fontsize=8)

    ax.set_ylabel('Normalized Value')
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Score')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        return None

    return fig


def marginal_plots(history: List[Dict],
                   title: str = "Marginal Plots",
                   save_path: Optional[str] = None,
                   show: bool = True) -> Optional[plt.Figure]:
    """
    Plot each parameter vs score in a grid.

    Shows the relationship between each individual parameter and the score.

    Args:
        history: List of trial results
        title: Plot title
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        matplotlib Figure if not showing, else None
    """
    _check_matplotlib()

    params_list, scores, param_names = _extract_data(history)

    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_params == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, param in enumerate(param_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        values = [p[param] for p in params_list]

        # Check if values are categorical or numeric
        unique_vals = sorted(set(values))
        if len(unique_vals) <= 10:
            # Treat as categorical - use box plot style
            val_to_scores = {v: [] for v in unique_vals}
            for v, s in zip(values, scores):
                val_to_scores[v].append(s)

            positions = range(len(unique_vals))
            box_data = [val_to_scores[v] for v in unique_vals]

            ax.boxplot(box_data, positions=positions)
            ax.set_xticks(positions)
            ax.set_xticklabels([str(v) for v in unique_vals], rotation=45, ha='right')

            # Also scatter the points
            for i, v in enumerate(unique_vals):
                jitter = np.random.uniform(-0.1, 0.1, len(val_to_scores[v]))
                ax.scatter([i + j for j in jitter], val_to_scores[v], alpha=0.5, s=20)
        else:
            # Numeric - scatter plot
            ax.scatter(values, scores, alpha=0.6)

        ax.set_xlabel(param)
        ax.set_ylabel('Score')
        ax.set_title(f'{param} vs Score')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_params, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        return None

    return fig


def heatmap(history: List[Dict],
            param1: str,
            param2: str,
            title: Optional[str] = None,
            aggregation: str = 'max',
            save_path: Optional[str] = None,
            show: bool = True) -> Optional[plt.Figure]:
    """
    Plot 2D heatmap of two parameters vs score.

    Args:
        history: List of trial results
        param1: First parameter name (x-axis)
        param2: Second parameter name (y-axis)
        title: Plot title
        aggregation: How to aggregate scores for same param combo ('max', 'mean')
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        matplotlib Figure if not showing, else None
    """
    _check_matplotlib()

    params_list, scores, _ = _extract_data(history)

    # Get unique values for each parameter
    vals1 = sorted(set(p[param1] for p in params_list))
    vals2 = sorted(set(p[param2] for p in params_list))

    # Create score matrix
    score_matrix = np.full((len(vals2), len(vals1)), np.nan)
    count_matrix = np.zeros((len(vals2), len(vals1)))

    val1_to_idx = {v: i for i, v in enumerate(vals1)}
    val2_to_idx = {v: i for i, v in enumerate(vals2)}

    for params, score in zip(params_list, scores):
        i = val2_to_idx[params[param2]]
        j = val1_to_idx[params[param1]]

        if np.isnan(score_matrix[i, j]):
            score_matrix[i, j] = score
            count_matrix[i, j] = 1
        else:
            if aggregation == 'max':
                score_matrix[i, j] = max(score_matrix[i, j], score)
            else:  # mean
                score_matrix[i, j] = (score_matrix[i, j] * count_matrix[i, j] + score) / (count_matrix[i, j] + 1)
            count_matrix[i, j] += 1

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    if HAS_SEABORN:
        sns.heatmap(score_matrix, ax=ax, annot=True, fmt='.3f',
                   xticklabels=[str(v) for v in vals1],
                   yticklabels=[str(v) for v in vals2],
                   cmap='viridis')
    else:
        im = ax.imshow(score_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(vals1)))
        ax.set_xticklabels([str(v) for v in vals1])
        ax.set_yticks(range(len(vals2)))
        ax.set_yticklabels([str(v) for v in vals2])
        plt.colorbar(im, ax=ax, label='Score')

        # Add text annotations
        for i in range(len(vals2)):
            for j in range(len(vals1)):
                if not np.isnan(score_matrix[i, j]):
                    ax.text(j, i, f'{score_matrix[i, j]:.3f}',
                           ha='center', va='center', fontsize=8)

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    if title is None:
        title = f'{param1} vs {param2} Heatmap'
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        return None

    return fig


def param_importance(history: List[Dict],
                     title: str = "Parameter Importance",
                     save_path: Optional[str] = None,
                     show: bool = True) -> Optional[plt.Figure]:
    """
    Plot parameter importance based on variance in scores.

    Uses a simple variance-based importance measure:
    For each parameter, calculates how much score variance is explained
    by grouping trials by that parameter value.

    Args:
        history: List of trial results
        title: Plot title
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        matplotlib Figure if not showing, else None
    """
    _check_matplotlib()

    params_list, scores, param_names = _extract_data(history)

    scores = np.array(scores)
    total_variance = np.var(scores) if len(scores) > 1 else 0

    importance = {}

    for param in param_names:
        values = [p[param] for p in params_list]
        unique_vals = list(set(values))

        if len(unique_vals) <= 1:
            importance[param] = 0
            continue

        # Group scores by parameter value
        groups = {v: [] for v in unique_vals}
        for v, s in zip(values, scores):
            groups[v].append(s)

        # Calculate between-group variance (simplified importance)
        group_means = [np.mean(groups[v]) for v in unique_vals if len(groups[v]) > 0]
        between_variance = np.var(group_means) if len(group_means) > 1 else 0

        # Importance as ratio of between-variance to total variance
        if total_variance > 0:
            importance[param] = between_variance / total_variance
        else:
            importance[param] = 0

    # Normalize importances
    total_importance = sum(importance.values())
    if total_importance > 0:
        importance = {k: v / total_importance for k, v in importance.items()}

    # Sort by importance
    sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_params]
    values = [p[1] for p in sorted_params]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors)

    ax.set_xlabel('Relative Importance')
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.1 if values else 1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.2%}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        return None

    return fig


def plot_all(history: List[Dict],
             save_dir: Optional[str] = None,
             show: bool = True,
             prefix: str = "hpo") -> Dict[str, plt.Figure]:
    """
    Generate all visualization plots for HPO results.

    Args:
        history: List of trial results
        save_dir: If provided, save all figures to this directory
        show: If True, display plots
        prefix: Prefix for saved file names

    Returns:
        Dict mapping plot names to Figure objects
    """
    _check_matplotlib()

    import os

    figures = {}

    # Convergence
    fig = convergence(history, show=False)
    figures['convergence'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'{prefix}_convergence.png'),
                   dpi=150, bbox_inches='tight')

    # Parallel coordinates
    fig = parallel_coordinates(history, show=False)
    figures['parallel_coordinates'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'{prefix}_parallel.png'),
                   dpi=150, bbox_inches='tight')

    # Marginal plots
    fig = marginal_plots(history, show=False)
    figures['marginal'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'{prefix}_marginal.png'),
                   dpi=150, bbox_inches='tight')

    # Parameter importance
    fig = param_importance(history, show=False)
    figures['importance'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'{prefix}_importance.png'),
                   dpi=150, bbox_inches='tight')

    # Generate heatmaps for all param pairs if feasible
    _, _, param_names = _extract_data(history)
    if len(param_names) >= 2:
        from itertools import combinations
        for p1, p2 in combinations(param_names, 2):
            fig = heatmap(history, p1, p2, show=False)
            key = f'heatmap_{p1}_{p2}'
            figures[key] = fig
            if save_dir:
                fig.savefig(os.path.join(save_dir, f'{prefix}_{key}.png'),
                           dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return figures
