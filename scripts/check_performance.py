# scripts/check_performance.py

import argparse
import json
import mlflow
from pathlib import Path
from datetime import datetime


def get_latest_run_metrics():
    """
    Récupère les métriques du dernier run MLflow.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        # Récupérer l'expérience
        experiment = mlflow.get_experiment_by_name("PredictiveMaintenance")
        
        if not experiment:
            print("[ERROR] Experiment 'PredictiveMaintenance' not found")
            return None
        
        # Récupérer tous les runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=2  # On prend les 2 derniers pour comparer
        )
        
        if runs.empty:
            print("[ERROR] No runs found")
            return None
        
        latest_run = runs.iloc[0]
        
        metrics = {
            "run_id": latest_run.get("run_id", "unknown"),
            "accuracy": float(latest_run.get("metrics.accuracy", 0)),
            "roc_auc": float(latest_run.get("metrics.roc_auc", 0)),
            "model_name": latest_run.get("params.model_name", "Unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Si on a un run précédent, comparer
        if len(runs) > 1:
            previous_run = runs.iloc[1]
            metrics["previous_auc"] = float(previous_run.get("metrics.roc_auc", 0))
            metrics["improvement"] = metrics["roc_auc"] - metrics["previous_auc"]
        
        return metrics
    
    except Exception as e:
        print(f"[ERROR] Failed to retrieve metrics: {e}")
        return None


def check_performance(threshold=0.85):
    """
    Vérifie si le modèle respecte les critères de performance.
    Returns 0 si succès, 1 si échec.
    """
    print("\n" + "="*60)
    print("  🔍 MODEL PERFORMANCE VALIDATION")
    print("="*60 + "\n")
    
    metrics = get_latest_run_metrics()
    
    if not metrics:
        print("❌ Could not retrieve metrics")
        return 1
    
    # Afficher les métriques
    print(f"📊 Latest Model Performance:")
    print(f"   Model: {metrics['model_name']}")
    print(f"   Run ID: {metrics['run_id'][:8]}...")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   Threshold: {threshold}")
    
    if "improvement" in metrics:
        improvement_symbol = "📈" if metrics["improvement"] > 0 else "📉"
        print(f"   {improvement_symbol} Change: {metrics['improvement']:+.4f}")
    
    print("\n" + "-"*60)
    
    # Vérification du seuil
    passed = metrics['roc_auc'] >= threshold
    
    if passed:
        print(f"\n✅ VALIDATION PASSED!")
        print(f"   AUC ({metrics['roc_auc']:.4f}) >= Threshold ({threshold})")
        status = "passed"
        exit_code = 0
    else:
        print(f"\n❌ VALIDATION FAILED!")
        print(f"   AUC ({metrics['roc_auc']:.4f}) < Threshold ({threshold})")
        status = "failed"
        exit_code = 1
    
    # Sauvegarder le rapport
    report = {
        "status": status,
        "metrics": metrics,
        "threshold": threshold,
        "validation_time": datetime.now().isoformat()
    }
    
    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports/last_training.json")
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path}")
    print("="*60 + "\n")
    
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate model performance against threshold"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.85, 
        help="Minimum ROC-AUC threshold (default: 0.85)"
    )
    args = parser.parse_args()
    
    exit_code = check_performance(args.threshold)
    exit(exit_code)