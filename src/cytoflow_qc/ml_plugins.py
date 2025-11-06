"""Machine learning-based plugins for cytoflow-qc."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd

# Try to import ML libraries
try:
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .plugins.gating import GatingStrategyPlugin
from .plugins.base import PluginError


class MLGatingStrategyPlugin(GatingStrategyPlugin):
    """Machine learning-based gating strategy plugin."""

    PLUGIN_TYPE = "ml_gating"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize ML gating plugin.

        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self._scaler: StandardScaler | None = None
        self._cluster_model: Any | None = None
        self._cluster_labels: np.ndarray | None = None

    def validate_gate_parameters(self, channels: dict[str, str]) -> None:
        """Validate required channels for ML gating."""
        required_channels = ["fsc_a", "ssc_a"]
        for channel in required_channels:
            if channel not in channels:
                raise ValueError(f"Required channel '{channel}' not found for ML gating")

    def get_gate_description(self) -> str:
        """Get description of ML gating strategy."""
        method = self.config.get("method", "hdbscan")
        return f"Machine learning-based gating using {method} clustering"

    def get_required_parameters(self) -> list[str]:
        """Get required configuration parameters."""
        return ["method", "channels"]

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for ML gating."""
        return {
            "method": "hdbscan",
            "min_cluster_size": 50,
            "min_samples": 10,
            "cluster_selection_epsilon": 0.1,
            "standardize": True,
            "n_clusters": None,  # For KMeans/GMM
            "random_state": 42,
        }

    def apply_gate(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply ML-based gating strategy.

        Args:
            data: Input DataFrame
            channels: Channel mapping
            **kwargs: Additional parameters

        Returns:
            Tuple of (gated_data, gating_params)
        """
        if not SKLEARN_AVAILABLE and not HDBSCAN_AVAILABLE:
            raise PluginError("ML libraries (scikit-learn or hdbscan) not available")

        # Extract relevant channels for clustering
        cluster_channels = []
        for channel_type in ["fsc_a", "ssc_a"]:
            channel_name = channels.get(channel_type)
            if channel_name and channel_name in data.columns:
                cluster_channels.append(channel_name)

        if len(cluster_channels) < 2:
            raise ValueError("Need at least 2 channels for ML gating")

        # Prepare data for clustering
        X = data[cluster_channels].values

        # Standardize if requested
        if self.config.get("standardize", True):
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        # Apply clustering based on method
        method = self.config.get("method", "hdbscan")

        if method == "hdbscan":
            cluster_labels, model = self._apply_hdbscan(X)
        elif method == "kmeans":
            cluster_labels, model = self._apply_kmeans(X)
        elif method == "gmm":
            cluster_labels, model = self._apply_gmm(X)
        elif method == "hierarchical":
            cluster_labels, model = self._apply_hierarchical(X)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        self._cluster_model = model
        self._cluster_labels = cluster_labels

        # Create gating mask (exclude noise points for HDBSCAN)
        if method == "hdbscan" and hasattr(model, 'labels_'):
            # For HDBSCAN, label -1 indicates noise
            mask = cluster_labels != -1
        else:
            # For other methods, include all points
            mask = np.ones(len(data), dtype=bool)

        # Apply mask to data
        gated_data = data[mask].copy()
        gated_data["cluster_label"] = cluster_labels[mask]

        # Calculate gating parameters
        gating_params = {
            "method": method,
            "n_clusters": len(np.unique(cluster_labels[mask])) if method != "hdbscan" else len(np.unique(cluster_labels[cluster_labels != -1])),
            "total_events": len(data),
            "gated_events": len(gated_data),
            "retention_rate": len(gated_data) / len(data),
            "cluster_sizes": dict(pd.Series(cluster_labels[mask]).value_counts()),
        }

        return gated_data, gating_params

    def _apply_hdbscan(self, X: np.ndarray) -> tuple[np.ndarray, Any]:
        """Apply HDBSCAN clustering."""
        if not HDBSCAN_AVAILABLE:
            raise PluginError("HDBSCAN not available")

        min_cluster_size = self.config.get("min_cluster_size", 50)
        min_samples = self.config.get("min_samples", 10)
        cluster_selection_epsilon = self.config.get("cluster_selection_epsilon", 0.1)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

        labels = clusterer.fit_predict(X)
        return labels, clusterer

    def _apply_kmeans(self, X: np.ndarray) -> tuple[np.ndarray, Any]:
        """Apply K-means clustering."""
        if not SKLEARN_AVAILABLE:
            raise PluginError("scikit-learn not available")

        n_clusters = self.config.get("n_clusters")
        if n_clusters is None:
            # Auto-determine number of clusters using silhouette score
            n_clusters = self._find_optimal_k(X)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.get("random_state", 42),
            n_init=10
        )

        labels = kmeans.fit_predict(X)
        return labels, kmeans

    def _apply_gmm(self, X: np.ndarray) -> tuple[np.ndarray, Any]:
        """Apply Gaussian Mixture Model clustering."""
        if not SKLEARN_AVAILABLE:
            raise PluginError("scikit-learn not available")

        n_components = self.config.get("n_clusters")
        if n_components is None:
            n_components = self._find_optimal_k(X)

        gmm = GaussianMixture(
            n_components=n_components,
            random_state=self.config.get("random_state", 42)
        )

        labels = gmm.fit_predict(X)
        return labels, gmm

    def _apply_hierarchical(self, X: np.ndarray) -> tuple[np.ndarray, Any]:
        """Apply hierarchical clustering."""
        if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
            raise PluginError("Required libraries not available")

        # Compute linkage matrix
        distance_matrix = pdist(X)
        linkage_matrix = linkage(distance_matrix, method='ward')

        n_clusters = self.config.get("n_clusters", 3)
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        return labels, linkage_matrix

    def _find_optimal_k(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using silhouette score."""
        if not SKLEARN_AVAILABLE:
            return 3  # Default fallback

        best_k = 2
        best_score = -1

        for k in range(2, min(max_k + 1, len(X) // 10)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception:
                continue

        return best_k


class AnomalyDetectionPlugin(GatingStrategyPlugin):
    """Anomaly detection plugin for identifying unusual samples."""

    PLUGIN_TYPE = "anomaly_detection"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize anomaly detection plugin."""
        super().__init__(config)
        self._isolation_forest: Any | None = None
        self._anomaly_scores: np.ndarray | None = None

    def validate_gate_parameters(self, channels: dict[str, str]) -> None:
        """Validate required channels for anomaly detection."""
        # Anomaly detection can work with any channels
        pass

    def get_gate_description(self) -> str:
        """Get description of anomaly detection strategy."""
        method = self.config.get("method", "isolation_forest")
        return f"Anomaly detection using {method}"

    def get_required_parameters(self) -> list[str]:
        """Get required configuration parameters."""
        return ["method", "contamination"]

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for anomaly detection."""
        return {
            "method": "isolation_forest",
            "contamination": 0.1,  # Expected proportion of anomalies
            "random_state": 42,
        }

    def apply_gate(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply anomaly detection gating.

        Args:
            data: Input DataFrame
            channels: Channel mapping
            **kwargs: Additional parameters

        Returns:
            Tuple of (gated_data, gating_params)
        """
        if not SKLEARN_AVAILABLE:
            raise PluginError("scikit-learn not available for anomaly detection")

        from sklearn.ensemble import IsolationForest

        # Prepare feature matrix
        feature_cols = [col for col in data.columns if col not in ["sample_id", "batch", "condition"]]
        X = data[feature_cols].values

        # Fit isolation forest
        contamination = self.config.get("contamination", 0.1)
        random_state = self.config.get("random_state", 42)

        self._isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

        # Predict anomalies (-1 for anomalies, 1 for normal)
        predictions = self._isolation_forest.fit_predict(X)
        self._anomaly_scores = self._isolation_forest.decision_function(X)

        # Keep only normal samples
        normal_mask = predictions == 1
        gated_data = data[normal_mask].copy()

        # Calculate gating parameters
        gating_params = {
            "method": "isolation_forest",
            "contamination": contamination,
            "total_samples": len(data),
            "normal_samples": len(gated_data),
            "anomaly_samples": len(data) - len(gated_data),
            "anomaly_fraction": (len(data) - len(gated_data)) / len(data),
        }

        return gated_data, gating_params


class BayesianOptimizationPlugin(GatingStrategyPlugin):
    """Bayesian optimization for automatic parameter tuning."""

    PLUGIN_TYPE = "bayesian_optimization"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Bayesian optimization plugin."""
        super().__init__(config)
        self._optimization_results: dict[str, Any] | None = None

    def validate_gate_parameters(self, channels: dict[str, str]) -> None:
        """Validate required channels for optimization."""
        pass

    def get_gate_description(self) -> str:
        """Get description of optimization strategy."""
        return "Bayesian optimization for automatic parameter tuning"

    def get_required_parameters(self) -> list[str]:
        """Get required configuration parameters."""
        return ["objective", "parameter_ranges"]

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for Bayesian optimization."""
        return {
            "objective": "silhouette_score",
            "parameter_ranges": {
                "min_cluster_size": [10, 100],
                "min_samples": [5, 20],
            },
            "n_iterations": 50,
            "random_state": 42,
        }

    def apply_gate(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply Bayesian optimization for parameter tuning.

        Args:
            data: Input DataFrame
            channels: Channel mapping
            **kwargs: Additional parameters

        Returns:
            Tuple of (gated_data, gating_params)
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Integer
            from skopt.utils import use_named_args
        except ImportError:
            raise PluginError("scikit-optimize not available for Bayesian optimization")

        # Define parameter space
        param_ranges = self.config.get("parameter_ranges", {})
        dimensions = []

        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                dimensions.append(Integer(low=min_val, high=max_val, name=param))

        if not dimensions:
            raise ValueError("No parameter ranges defined for optimization")

        # Define objective function
        objective = self.config.get("objective", "silhouette_score")

        @use_named_args(dimensions)
        def objective_function(**params):
            """Objective function for optimization."""
            try:
                # Create temporary config with optimized parameters
                temp_config = self.config.copy()
                temp_config.update(params)

                # Create ML gating plugin with optimized parameters
                ml_plugin = MLGatingStrategyPlugin(temp_config)
                gated_data, _ = ml_plugin.apply_gate(data, channels)

                if len(gated_data) == 0:
                    return 1.0  # Bad result

                # Calculate objective score
                if objective == "silhouette_score":
                    if SKLEARN_AVAILABLE:
                        score = silhouette_score(
                            gated_data[["FSC-A", "SSC-A"]].values,
                            gated_data.get("cluster_label", np.zeros(len(gated_data)))
                        )
                        return -score  # Minimize negative score
                elif objective == "retention_rate":
                    return -len(gated_data) / len(data)  # Minimize negative retention

                return 0.0  # Default

            except Exception:
                return 1.0  # Bad result

        # Run optimization
        n_iterations = self.config.get("n_iterations", 50)
        result = gp_minimize(
            objective_function,
            dimensions,
            n_calls=n_iterations,
            random_state=self.config.get("random_state", 42)
        )

        # Get best parameters
        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        best_score = -result.fun  # Convert back from minimization

        self._optimization_results = {
            "best_params": best_params,
            "best_score": best_score,
            "all_scores": [-f for f in result.func_vals],
        }

        # Apply gating with optimized parameters
        optimized_config = self.config.copy()
        optimized_config.update(best_params)

        ml_plugin = MLGatingStrategyPlugin(optimized_config)
        gated_data, gating_params = ml_plugin.apply_gate(data, channels)

        # Add optimization results to parameters
        gating_params.update({
            "optimization_method": "bayesian",
            "best_parameters": best_params,
            "optimization_score": best_score,
        })

        return gated_data, gating_params


class TransferLearningPlugin(GatingStrategyPlugin):
    """Transfer learning plugin for pre-trained gating models."""

    PLUGIN_TYPE = "transfer_learning"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize transfer learning plugin."""
        super().__init__(config)
        self._pretrained_model: Any | None = None

    def validate_gate_parameters(self, channels: dict[str, str]) -> None:
        """Validate required channels for transfer learning."""
        pass

    def get_gate_description(self) -> str:
        """Get description of transfer learning strategy."""
        cell_type = self.config.get("cell_type", "generic")
        return f"Transfer learning-based gating for {cell_type} cells"

    def get_required_parameters(self) -> list[str]:
        """Get required configuration parameters."""
        return ["cell_type", "model_path"]

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for transfer learning."""
        return {
            "cell_type": "generic",
            "model_path": None,  # Path to pre-trained model
            "fine_tune": True,
            "adaptation_rate": 0.1,
        }

    def apply_gate(
        self,
        data: pd.DataFrame,
        channels: dict[str, str],
        **kwargs
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply transfer learning-based gating.

        Args:
            data: Input DataFrame
            channels: Channel mapping
            **kwargs: Additional parameters

        Returns:
            Tuple of (gated_data, gating_params)
        """
        # This is a simplified implementation
        # In practice, this would load pre-trained models and adapt them

        cell_type = self.config.get("cell_type", "generic")

        # For now, fall back to standard ML gating
        # In a real implementation, this would:
        # 1. Load pre-trained model for the specified cell type
        # 2. Adapt the model to the current data distribution
        # 3. Apply the adapted model for gating

        # Placeholder implementation
        from .plugins.gating import GatingStrategyPlugin

        # Use ML gating as fallback
        ml_config = self.config.copy()
        ml_config["method"] = "hdbscan"

        ml_plugin = MLGatingStrategyPlugin(ml_config)
        gated_data, gating_params = ml_plugin.apply_gate(data, channels)

        gating_params.update({
            "method": "transfer_learning",
            "cell_type": cell_type,
            "adaptation_applied": True,
        })

        return gated_data, gating_params


# Example pre-defined ML gating strategies
class LymphocyteGatingPlugin(MLGatingStrategyPlugin):
    """Specialized ML gating for lymphocyte populations."""

    PLUGIN_NAME = "lymphocyte_gating"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "ML-based gating optimized for lymphocyte populations"

    def get_default_config(self) -> dict[str, Any]:
        """Get optimized configuration for lymphocyte gating."""
        base_config = super().get_default_config()
        base_config.update({
            "method": "hdbscan",
            "min_cluster_size": 100,
            "min_samples": 15,
        })
        return base_config


class StemCellGatingPlugin(MLGatingStrategyPlugin):
    """Specialized ML gating for stem cell populations."""

    PLUGIN_NAME = "stem_cell_gating"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "ML-based gating optimized for stem cell populations"

    def get_default_config(self) -> dict[str, Any]:
        """Get optimized configuration for stem cell gating."""
        base_config = super().get_default_config()
        base_config.update({
            "method": "gmm",
            "n_clusters": 3,  # Typically 3 populations for stem cells
        })
        return base_config


# Auto-register plugins
def register_ml_plugins() -> None:
    """Register ML plugins with the plugin registry."""
    from .plugins import get_plugin_registry

    registry = get_plugin_registry()

    # Register ML gating strategies
    registry._plugins["gating_strategy"]["lymphocyte_gating"] = LymphocyteGatingPlugin
    registry._plugins["gating_strategy"]["stem_cell_gating"] = StemCellGatingPlugin

    # Register anomaly detection
    registry._plugins["gating_strategy"]["anomaly_detection"] = AnomalyDetectionPlugin

    # Register optimization
    registry._plugins["gating_strategy"]["bayesian_optimization"] = BayesianOptimizationPlugin

    # Register transfer learning
    registry._plugins["gating_strategy"]["transfer_learning"] = TransferLearningPlugin


# Call registration when module is imported
register_ml_plugins()
















