"""
Week 3: Visualization Generation
Creates plots and figures for the research paper
"""

import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluator import HallucinationEvaluator

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_plotting_style():
    """Set up matplotlib style for publication-quality figures"""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")

    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 300

def plot_performance_vs_quality(metrics_df: pd.DataFrame, config, metric='f1'):
    """
    Plot detector performance vs retrieval quality

    Creates a line plot showing how each detector's performance
    changes across quality tiers
    """
    print(f"Creating performance vs quality plot ({metric})...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define quality tier ordering
    tier_order = ['high', 'medium', 'low']
    tier_mapping = {tier: i for i, tier in enumerate(tier_order)}

    # Plot each detector
    detectors = metrics_df['detector'].unique()

    for detector in detectors:
        detector_data = metrics_df[metrics_df['detector'] == detector].copy()
        detector_data['tier_num'] = detector_data['quality_tier'].map(tier_mapping)
        detector_data = detector_data.sort_values('tier_num')

        ax.plot(
            detector_data['quality_tier'],
            detector_data[metric],
            marker='o',
            linewidth=2,
            label=detector,
            markersize=8
        )

    ax.set_xlabel('Retrieval Quality Tier', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
    ax.set_title(f'Detector Performance vs Retrieval Quality ({metric.upper()})',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Detector', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save
    output_path = Path(config['paths']['visualizations_dir']) / f'performance_vs_quality_{metric}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

    plt.close()

def plot_all_metrics_comparison(metrics_df: pd.DataFrame, config):
    """
    Create a comprehensive comparison of all metrics across tiers
    """
    print("Creating comprehensive metrics comparison...")

    metrics = ['precision', 'recall', 'f1', 'auroc']
    tier_order = ['high', 'medium', 'low']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare data for grouped bar chart
        pivot_data = metrics_df.pivot(
            index='quality_tier',
            columns='detector',
            values=metric
        ).reindex(tier_order)

        pivot_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_xlabel('Retrieval Quality Tier', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        ax.legend(title='Detector', frameon=True)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(tier_order, rotation=0)

    plt.tight_layout()

    output_path = Path(config['paths']['visualizations_dir']) / 'all_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

    plt.close()

def plot_confusion_matrices(config):
    """
    Create confusion matrices for all detectors
    """
    print("Creating confusion matrices...")

    evaluator = HallucinationEvaluator(config)

    for tier in ['high', 'medium', 'low']:
        try:
            cms = evaluator.compute_confusion_matrices(tier)

            if not cms:
                continue

            fig, axes = plt.subplots(1, len(cms), figsize=(5*len(cms), 4))

            if len(cms) == 1:
                axes = [axes]

            for idx, (detector_name, cm) in enumerate(cms.items()):
                ax = axes[idx]

                # Plot confusion matrix
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax,
                    cbar=True,
                    square=True
                )

                ax.set_xlabel('Predicted', fontweight='bold')
                ax.set_ylabel('True', fontweight='bold')
                ax.set_title(f'{detector_name}\n({tier.capitalize()} Quality)',
                           fontweight='bold')
                ax.set_xticklabels(['Faithful', 'Hallucinated'])
                ax.set_yticklabels(['Faithful', 'Hallucinated'], rotation=0)

            plt.tight_layout()

            output_path = Path(config['paths']['visualizations_dir']) / f'confusion_matrix_{tier}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

            plt.close()

        except Exception as e:
            print(f"Warning: Could not create confusion matrix for {tier}: {e}")

def plot_retrieval_quality_distribution(config):
    """
    Visualize the distribution of relevant vs distractor documents
    """
    print("Creating retrieval quality distribution plot...")

    tiers = ['high', 'medium', 'low']
    tier_config = config['retrieval']['quality_tiers']

    data = []
    for tier in tiers:
        data.append({
            'tier': tier.capitalize(),
            'Relevant': tier_config[tier]['relevant_ratio'] * 100,
            'Distractors': tier_config[tier]['distractor_ratio'] * 100
        })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tiers))
    width = 0.35

    ax.bar(x - width/2, df['Relevant'], width, label='Relevant', color='#2ecc71')
    ax.bar(x + width/2, df['Distractors'], width, label='Distractors', color='#e74c3c')

    ax.set_xlabel('Retrieval Quality Tier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Retrieval Quality Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['tier'])
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for i, tier in enumerate(tiers):
        ax.text(i - width/2, df.loc[i, 'Relevant'] + 2,
               f"{df.loc[i, 'Relevant']:.0f}%",
               ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, df.loc[i, 'Distractors'] + 2,
               f"{df.loc[i, 'Distractors']:.0f}%",
               ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    output_path = Path(config['paths']['visualizations_dir']) / 'retrieval_quality_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

    plt.close()

def create_results_table(metrics_df: pd.DataFrame, config):
    """
    Create a formatted table of results for the paper
    """
    print("Creating results table...")

    # Pivot for better readability
    table_data = []

    for detector in metrics_df['detector'].unique():
        detector_data = metrics_df[metrics_df['detector'] == detector]

        for tier in ['high', 'medium', 'low']:
            tier_data = detector_data[detector_data['quality_tier'] == tier]

            if len(tier_data) > 0:
                row = tier_data.iloc[0]
                table_data.append({
                    'Detector': detector,
                    'Quality': tier.capitalize(),
                    'Precision': f"{row['precision']:.3f}",
                    'Recall': f"{row['recall']:.3f}",
                    'F1': f"{row['f1']:.3f}",
                    'AUROC': f"{row['auroc']:.3f}",
                    'Samples': int(row['num_samples'])
                })

    results_table = pd.DataFrame(table_data)

    # Save as CSV
    output_path = Path(config['paths']['results_dir']) / 'results_table.csv'
    results_table.to_csv(output_path, index=False)
    print(f"Saved table to {output_path}")

    # Also save as LaTeX for paper
    latex_path = Path(config['paths']['results_dir']) / 'results_table.tex'
    with open(latex_path, 'w') as f:
        f.write(results_table.to_latex(index=False))
    print(f"Saved LaTeX table to {latex_path}")

    # Print table
    print("\nResults Table:")
    print(results_table.to_string(index=False))

def plot_auroc_heatmap(metrics_df: pd.DataFrame, config):
    """
    Create a heatmap of AUROC scores
    """
    print("Creating AUROC heatmap...")

    # Pivot data
    pivot_data = metrics_df.pivot(
        index='detector',
        columns='quality_tier',
        values='auroc'
    )[['high', 'medium', 'low']]

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'AUROC'},
        linewidths=1,
        linecolor='white'
    )

    ax.set_xlabel('Retrieval Quality Tier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detector', fontsize=12, fontweight='bold')
    ax.set_title('AUROC Scores Across Quality Tiers', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['High', 'Medium', 'Low'], rotation=0)

    plt.tight_layout()

    output_path = Path(config['paths']['visualizations_dir']) / 'auroc_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

    plt.close()

def main():
    """Main visualization function"""
    config = load_config()

    print("="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Set up plotting style
    setup_plotting_style()

    # Load evaluation metrics
    evaluator = HallucinationEvaluator(config)
    metrics_df = evaluator.evaluate_all_tiers()

    if len(metrics_df) == 0:
        print("No evaluation results found. Run evaluation first.")
        return

    # Create all visualizations
    print("\n1. Performance vs Quality Plots")
    for metric in ['precision', 'recall', 'f1', 'auroc']:
        plot_performance_vs_quality(metrics_df, config, metric)

    print("\n2. All Metrics Comparison")
    plot_all_metrics_comparison(metrics_df, config)

    print("\n3. Confusion Matrices")
    plot_confusion_matrices(config)

    print("\n4. Retrieval Quality Distribution")
    plot_retrieval_quality_distribution(config)

    print("\n5. AUROC Heatmap")
    plot_auroc_heatmap(metrics_df, config)

    print("\n6. Results Table")
    create_results_table(metrics_df, config)

    print("\n" + "="*60)
    print("VISUALIZATIONS COMPLETE!")
    print(f"Saved to: {config['paths']['visualizations_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()
