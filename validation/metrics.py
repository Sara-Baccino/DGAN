from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict
import numpy as np

class SyntheticDataEvaluator:
    """
    Valutazione completa dati sintetici:
    1. Utility metrics (statistical fidelity)
    2. Privacy metrics (disclosure risk)
    """
    
    def __init__(self, real_data: Dict[str, np.ndarray], synthetic_data: Dict[str, np.ndarray]):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.results = {}
    
    def evaluate_all(self) -> Dict:
        """Esegue tutte le valutazioni."""
        print("Running evaluation...")
        
        self.results['univariate'] = self.evaluate_univariate()
        self.results['correlations'] = self.evaluate_correlations()
        self.results['temporal'] = self.evaluate_temporal_patterns()
        self.results['privacy'] = self.evaluate_privacy()
        
        return self.results
    
    def evaluate_univariate(self) -> Dict:
        """Confronto distribuzioni univariate."""
        metrics = {}
        
        for var_name in self.real_data.keys():
            real_vals = self.real_data[var_name]
            syn_vals = self.synthetic_data[var_name]
            
            # Rimuovi NaN
            real_flat = real_vals[~np.isnan(real_vals)].flatten()
            syn_flat = syn_vals[~np.isnan(syn_vals)].flatten()
            
            if len(real_flat) == 0 or len(syn_flat) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(real_flat, syn_flat)
            
            # Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(real_flat, syn_flat)
            
            metrics[var_name] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'wasserstein_distance': float(wasserstein_dist),
                'real_mean': float(np.mean(real_flat)),
                'syn_mean': float(np.mean(syn_flat)),
                'real_std': float(np.std(real_flat)),
                'syn_std': float(np.std(syn_flat))
            }
        
        return metrics
    
    def evaluate_correlations(self) -> Dict:
        """Confronto matrici di correlazione."""
        # Prepara dati per correlazione (solo continuous)
        real_cont = []
        syn_cont = []
        var_names = []
        
        for var_name, var_data in self.real_data.items():
            if len(var_data.shape) == 1:  # Static
                real_flat = var_data[~np.isnan(var_data)]
                syn_flat = self.synthetic_data[var_name][~np.isnan(self.synthetic_data[var_name])]
                
                if len(real_flat) > 0 and len(syn_flat) > 0:
                    # Verifica se numeric
                    if np.issubdtype(real_flat.dtype, np.number):
                        real_cont.append(real_flat)
                        syn_cont.append(syn_flat)
                        var_names.append(var_name)
        
        if len(real_cont) < 2:
            return {}
        
        # Assicura stessa lunghezza
        min_len = min(len(x) for x in real_cont + syn_cont)
        real_cont = [x[:min_len] for x in real_cont]
        syn_cont = [x[:min_len] for x in syn_cont]
        
        real_matrix = np.column_stack(real_cont)
        syn_matrix = np.column_stack(syn_cont)
        
        real_corr = np.corrcoef(real_matrix.T)
        syn_corr = np.corrcoef(syn_matrix.T)
        
        # Frobenius norm della differenza
        corr_diff = np.linalg.norm(real_corr - syn_corr, 'fro')
        
        return {
            'correlation_difference': float(corr_diff),
            'variables': var_names
        }
    
    def evaluate_temporal_patterns(self) -> Dict:
        """Valuta pattern temporali (autocorrelazione, trend)."""
        metrics = {}
        
        for var_name, var_data in self.real_data.items():
            if len(var_data.shape) != 2:  # Solo temporal
                continue
            
            real_vals = var_data
            syn_vals = self.synthetic_data[var_name]
            
            # Autocorrelazione media
            real_autocorr = []