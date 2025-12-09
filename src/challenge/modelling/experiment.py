from __future__ import annotations
import pandas as pd
from copy import deepcopy
from typing import Dict, Any, Tuple, List, Optional, Union
from challenge.modelling.train_eval import cv_cost

from sklearn.pipeline import Pipeline

def run_experiment_grid(
    models: Dict[str, Any],
    feature_sets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    samplers: Dict[str, Optional[str]],
    tuning_strategies: Dict[str, bool],
    sampling_percentages: List[float],
    y_train: pd.Series,
    selectors: Optional[Dict[str, Any]] = None,
    n_cv_splits: int = 4,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Runs a grid of experiments over models, feature sets, samplers, tuning strategies, AND selectors.
    
    Args:
        models: Dictionary of model names to model objects.
        feature_sets: Dictionary of feature set names to (X_train, X_test) tuples.
        samplers: Dictionary of sampler names to sampler keys (e.g., 'smote', 'copula', None).
        tuning_strategies: Dictionary of tuning strategy names to boolean (True for tuned threshold).
        sampling_percentages: List of sampling percentages to test for samplers.
        y_train: Training labels.
        selectors: Dictionary of selector names to selector objects (e.g. {'Kruskal': KruskalSelector()}). 
                   If None, defaults to {'None': None}.
        n_cv_splits: Number of CV splits.
        random_state: Random state.
        verbose: Whether to print progress.
        
    Returns:
        pd.DataFrame containing the results of all experiments.
    """
    results_list = []
    
    # Default selectors if not provided
    if selectors is None:
        selectors = {'None': None}
        
    if verbose:
        print("--- STARTING EXPERIMENT MATRIX ---")
        
    # Outer loop: Iterate over models
    for model_name, model_obj in models.items():
        for fset_name, feature_data in feature_sets.items():
            if isinstance(feature_data, tuple):
                X_train_data, _ = feature_data
            else:
                X_train_data = feature_data
                
            for selector_name, selector_obj in selectors.items():
                
                for sampler_name, sampler_key in samplers.items():
                    
                    # Define the configurations to run for this sampler
                    configs = []
                    if sampler_key is not None:
                        # If sampler is used, iterate over percentages
                        for pct in sampling_percentages:
                            configs.append({'pct': pct, 'pct_name': f" | Sampling Percentage: {pct}"})
                    else:
                        # If no sampler, run once with pct=None
                        configs.append({'pct': None, 'pct_name': ""})
                        
                    for config in configs:
                        sampling_percentage = config['pct']
                        pct_name_suffix = config['pct_name']
                        
                        for tuning_name, tuning_bool in tuning_strategies.items():
                            feature_part_name = f"{fset_name}"
                            if selector_name != 'None':
                                feature_part_name += f" + {selector_name}"
                                
                            run_name = f"{model_name} | {feature_part_name} | {sampler_name} | {tuning_name}{pct_name_suffix}"
                            
                            if verbose:
                                print(f"\n=====================================")
                                print(f"RUNNING: {run_name}")
                                print(f"=====================================")
                            
                            # Build the dynamic pipeline or use model directly
                            # NOTE: cv_cost already includes Scaling and Balancing.
                            # We only need to bundle Selector + Model.
                            if selector_obj is not None:
                                # Create a pipeline: Selector -> Model
                                # Clone objects to ensure isolation
                                pipe_steps = [
                                    ('selector', deepcopy(selector_obj)),
                                    ('model', deepcopy(model_obj))
                                ]
                                model_to_run = Pipeline(pipe_steps)
                            else:
                                model_to_run = deepcopy(model_obj)
                            
                            # Prepare arguments for cv_cost
                            cv_kwargs = {
                                'model': model_to_run,
                                'X': X_train_data,
                                'y': y_train,
                                'sampler': sampler_key,
                                'tune_threshold': tuning_bool,
                                'folds': n_cv_splits,
                                'show_progress': True,
                                'verbose': True,
                                'random_state': random_state,
                            }
                            
                            # Only add sampling_strategy if a sampler is used
                            if sampler_key is not None:
                                cv_kwargs['sampling_strategy'] = sampling_percentage
                            
                            # Run Cross-Validation
                            try:
                                cv_results = cv_cost(**cv_kwargs)
                                
                                # Store results
                                result_entry = {
                                    'model': model_name,
                                    'feature_set': fset_name,
                                    'selector': selector_name,
                                    'sampler': sampler_name,
                                    'tuning': tuning_name,
                                    'cost_mean': cv_results['Cost_mean'],
                                    'cost_std': cv_results['Cost_std'],
                                    'auc_mean': cv_results['AUC_mean'],
                                    'f1_mean': cv_results['F1_mean'],
                                    'fit_time_mean': cv_results['fit_time_mean'],
                                    'run_name': run_name
                                }
                                
                                if sampling_percentage is not None:
                                    result_entry['sampling_percentage'] = sampling_percentage
                                    
                                results_list.append(result_entry)
                                
                            except Exception as e:
                                print(f"Error running {run_name}: {e}")
                                # raise e # Uncomment to debug
                                
    if verbose:
        print("\n--- EXPERIMENT MATRIX COMPLETE ---")
        
    return pd.DataFrame(results_list)
