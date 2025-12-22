#!/usr/bin/env python
"""
Quick Runner Script for C. elegans Cell Identification Evaluation Suite

This script provides simplified interfaces to run different evaluation scenarios:

Usage:
    # Full evaluation (no ablation retraining)
    python run_evaluation.py --mode full

    # Quick evaluation (core metrics only, no baselines)
    python run_evaluation.py --mode quick

    # Baselines only
    python run_evaluation.py --mode baselines

    # Robustness testing only
    python run_evaluation.py --mode robustness

    # Run ablation training (long-running)
    python run_evaluation.py --mode ablations --train-ablations

    # Generate figures only (requires prior run)
    python run_evaluation.py --mode figures --results-path evaluation_results/evaluation_results.json

Author: Generated for Henry Xue's research
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_suite import (
    EvaluationEngine,
    EvaluationResults,
    ConfidenceInterval,
    DataPaths,
    set_seed
)


def run_full_evaluation(args):
    """Run complete evaluation pipeline"""
    print("\n" + "="*70)
    print("SPARSE MATCHING EVALUATION SUITE - FULL EVALUATION")
    print("="*70)

    engine = EvaluationEngine(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        real_data_path=args.real_data,
        train_data_path=args.train_data,
        k_neighbors=args.k_neighbors,
        n_bootstrap=args.n_bootstrap
    )

    results = engine.run_full_evaluation(
        run_baselines=not args.no_baselines,
        run_ablations=args.run_ablations,
        output_dir=args.output_dir
    )

    return results


def run_quick_evaluation(args):
    """Run quick evaluation - core metrics only"""
    print("\n" + "="*70)
    print("SPARSE MATCHING EVALUATION SUITE - QUICK MODE")
    print("="*70)

    engine = EvaluationEngine(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        real_data_path=args.real_data,
        train_data_path=args.train_data,
        k_neighbors=args.k_neighbors,
        n_bootstrap=100  # Fewer bootstrap samples for speed
    )

    # Load and setup
    engine.load_data()
    engine.load_model()
    engine.build_reference_manifold(engine.train_data, max_samples=10000)

    # Evaluate only on simulated data
    print("\nRunning quick evaluation on simulated data...")
    sim_acc, sim_details = engine.evaluate_accuracy(engine.eval_data, "Quick eval")

    from evaluation_suite import StatisticalAnalysis
    engine.results.simulated_accuracy = StatisticalAnalysis.bootstrap_ci(sim_acc, 100)

    print(f"\n  Simulated accuracy: {engine.results.simulated_accuracy}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    engine.results.save_json(os.path.join(args.output_dir, "quick_results.json"))

    print(f"\nResults saved to: {args.output_dir}/quick_results.json")

    return engine.results


def run_baselines_only(args):
    """Run only baseline comparisons"""
    print("\n" + "="*70)
    print("SPARSE MATCHING EVALUATION SUITE - BASELINES ONLY")
    print("="*70)

    engine = EvaluationEngine(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        real_data_path=args.real_data,
        train_data_path=args.train_data,
        k_neighbors=args.k_neighbors,
        n_bootstrap=args.n_bootstrap
    )

    engine.load_data()
    engine.load_model()
    engine.build_reference_manifold(engine.train_data)

    # Run baseline evaluations
    from evaluation_suite import (
        ICPBaseline, CPDBaseline, HungarianBaseline,
        SiameseTransformer, StatisticalAnalysis, device
    )

    print("\n--- BASELINE COMPARISONS ---")

    # ICP
    print("\nEvaluating ICP...")
    icp = ICPBaseline()
    icp_acc = engine.evaluate_baseline(icp, engine.eval_data, "ICP")
    engine.results.icp_accuracy = StatisticalAnalysis.bootstrap_ci(icp_acc)
    print(f"  ICP: {engine.results.icp_accuracy}")

    # CPD
    print("\nEvaluating CPD...")
    cpd = CPDBaseline()
    cpd_acc = engine.evaluate_baseline(cpd, engine.eval_data, "CPD")
    engine.results.cpd_accuracy = StatisticalAnalysis.bootstrap_ci(cpd_acc)
    print(f"  CPD: {engine.results.cpd_accuracy}")

    # Hungarian
    print("\nEvaluating Hungarian...")
    hungarian = HungarianBaseline()
    hungarian_acc = engine.evaluate_baseline(hungarian, engine.eval_data, "Hungarian")
    engine.results.hungarian_accuracy = StatisticalAnalysis.bootstrap_ci(hungarian_acc)
    print(f"  Hungarian: {engine.results.hungarian_accuracy}")

    # Our method
    print("\nEvaluating Joint Attention (Ours)...")
    sim_acc, _ = engine.evaluate_accuracy(engine.eval_data, "Joint Attention")
    engine.results.joint_attention_accuracy = StatisticalAnalysis.bootstrap_ci(sim_acc)
    print(f"  Joint Attention: {engine.results.joint_attention_accuracy}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    engine.results.save_json(os.path.join(args.output_dir, "baselines_results.json"))

    csv_dir = os.path.join(args.output_dir, "csv_reports")
    engine.results.save_csv_reports(csv_dir)

    # Generate baseline comparison figure
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    engine._generate_figure3(figures_dir)

    return engine.results


def run_robustness_only(args):
    """Run only robustness testing"""
    print("\n" + "="*70)
    print("SPARSE MATCHING EVALUATION SUITE - ROBUSTNESS TESTING")
    print("="*70)

    engine = EvaluationEngine(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        real_data_path=args.real_data,
        train_data_path=args.train_data,
        k_neighbors=args.k_neighbors,
        n_bootstrap=args.n_bootstrap
    )

    engine.load_data()
    engine.load_model()
    engine.build_reference_manifold(engine.train_data)

    from evaluation_suite import StatisticalAnalysis

    print("\n--- ROBUSTNESS TO MISSING CELLS ---")
    missing_fractions = [0.0, 0.1, 0.2, 0.3, 0.5]

    for frac in missing_fractions:
        acc = engine.evaluate_with_perturbations(engine.eval_data, missing_fraction=frac)
        ci = StatisticalAnalysis.bootstrap_ci(acc, engine.n_bootstrap)
        print(f"  {frac:.0%} missing: {ci}")

        if frac == 0.0:
            engine.results.missing_0_accuracy = ci
        elif frac == 0.1:
            engine.results.missing_10_accuracy = ci
        elif frac == 0.2:
            engine.results.missing_20_accuracy = ci
        elif frac == 0.3:
            engine.results.missing_30_accuracy = ci
        elif frac == 0.5:
            engine.results.missing_50_accuracy = ci

    print("\n--- ROBUSTNESS TO COORDINATE NOISE ---")
    noise_scales = [0.0, 0.1, 0.2, 0.3, 0.5]

    for scale in noise_scales:
        acc = engine.evaluate_with_perturbations(engine.eval_data, noise_scale=scale)
        ci = StatisticalAnalysis.bootstrap_ci(acc, engine.n_bootstrap)
        print(f"  {scale}× NN distance: {ci}")

        if scale == 0.0:
            engine.results.noise_0_accuracy = ci
        elif scale == 0.1:
            engine.results.noise_01_accuracy = ci
        elif scale == 0.2:
            engine.results.noise_02_accuracy = ci
        elif scale == 0.3:
            engine.results.noise_03_accuracy = ci
        elif scale == 0.5:
            engine.results.noise_05_accuracy = ci

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    engine.results.save_json(os.path.join(args.output_dir, "robustness_results.json"))

    csv_dir = os.path.join(args.output_dir, "csv_reports")
    engine.results.save_csv_reports(csv_dir)

    # Generate robustness figure
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    engine._generate_figure4(figures_dir)

    return engine.results


def run_ablation_training(args):
    """Run ablation model training"""
    print("\n" + "="*70)
    print("ABLATION TRAINING")
    print("WARNING: This will take a very long time!")
    print("="*70)

    from ablation_trainer import AblationTrainer

    trainer = AblationTrainer(
        data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=os.path.join(args.output_dir, "ablation_models"),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=15,
        seed=42
    )

    if args.ablation:
        # Train specific ablation
        trainer.train_ablation(args.ablation)
    else:
        # Train all ablations
        trainer.train_all_ablations()


def regenerate_figures(args):
    """Regenerate figures from saved results"""
    print("\n" + "="*70)
    print("REGENERATING FIGURES")
    print("="*70)

    if not args.results_path or not os.path.exists(args.results_path):
        print("Error: --results-path required and must exist")
        return

    # Load results
    with open(args.results_path, 'r') as f:
        data = json.load(f)

    # Reconstruct EvaluationResults
    results = EvaluationResults()
    for key, value in data.items():
        if isinstance(value, dict) and 'mean' in value:
            setattr(results, key, ConfidenceInterval(
                mean=value['mean'],
                lower=value['lower'],
                upper=value['upper'],
                std=value.get('std', 0.0)
            ))
        else:
            setattr(results, key, value)

    # Create engine just for figure generation
    class FigureGenerator:
        def __init__(self, results):
            self.results = results

    engine = EvaluationEngine.__new__(EvaluationEngine)
    engine.results = results

    # Generate figures
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Generating figures to: {figures_dir}")
    engine._generate_figure2(figures_dir)
    engine._generate_figure3(figures_dir)
    engine._generate_figure4(figures_dir)
    engine._generate_figure5(figures_dir)
    engine._generate_figure6(figures_dir)
    engine._generate_figure7(figures_dir)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='C. elegans Cell Identification Evaluation Suite Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --mode full
  python run_evaluation.py --mode quick
  python run_evaluation.py --mode baselines
  python run_evaluation.py --mode robustness
  python run_evaluation.py --mode ablations --train-ablations
  python run_evaluation.py --mode figures --results-path results.json
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'quick', 'baselines', 'robustness', 'ablations', 'figures'],
                       help='Evaluation mode')

    # Data paths
    parser.add_argument('--model-path', type=str, default=DataPaths.MODEL_WEIGHTS,
                       help='Path to trained model weights')
    parser.add_argument('--eval-data', type=str, default=DataPaths.EVAL_DATA,
                       help='Path to evaluation data')
    parser.add_argument('--real-data', type=str, default=DataPaths.REAL_DATA,
                       help='Path to real embryo data')
    parser.add_argument('--train-data', type=str, default=DataPaths.TRAIN_DATA,
                       help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')

    # Evaluation options
    parser.add_argument('--no-baselines', action='store_true',
                       help='Skip baseline comparisons in full mode')
    parser.add_argument('--run-ablations', action='store_true',
                       help='Run ablation studies in full mode')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                       help='Number of bootstrap resamples')
    parser.add_argument('--k-neighbors', type=int, default=30,
                       help='k for k-NN classification')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Ablation training options
    parser.add_argument('--train-ablations', action='store_true',
                       help='Train ablation models (for --mode ablations)')
    parser.add_argument('--ablation', type=str,
                       choices=['full', 'raw_xyz', 'no_rel_pos', 'no_density',
                               'no_count', 'no_centroid', 'siamese', 'no_match_token'],
                       help='Specific ablation to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for ablations')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for ablation training')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate for ablation training')

    # Figure regeneration
    parser.add_argument('--results-path', type=str,
                       help='Path to saved results JSON for figure regeneration')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Run appropriate mode
    if args.mode == 'full':
        run_full_evaluation(args)
    elif args.mode == 'quick':
        run_quick_evaluation(args)
    elif args.mode == 'baselines':
        run_baselines_only(args)
    elif args.mode == 'robustness':
        run_robustness_only(args)
    elif args.mode == 'ablations':
        if args.train_ablations:
            run_ablation_training(args)
        else:
            print("Note: Use --train-ablations to actually train ablation models")
            print("This will take a very long time (hours to days depending on hardware)")
    elif args.mode == 'figures':
        regenerate_figures(args)


if __name__ == "__main__":
    main()
