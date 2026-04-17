class PhysicsModel:
    """
    First-order memory and compute approximation model for TinyML hardware.
    Enables quantifiable systems-level evaluation of edge constraints.
    """
    def __init__(self, bit_width=8, sparsity_gamma=0.5, overhead_factor=1.2):
        self.bit_width = bit_width
        self.byte_per_val = bit_width // 8
        self.gamma = sparsity_gamma  # Approximate tree sparsity after pruning
        self.overhead = overhead_factor # Streaming buffer and feature extraction overhead

    def calculate_sram_kb(self, n_saved_windows, feature_dim):
        """
        SRAM approximation incorporating fixed feature storage and dynamic streaming overhead.
        """
        base_bytes = n_saved_windows * feature_dim * self.byte_per_val
        total_bytes = base_bytes * self.overhead
        return total_bytes / 1024.0

    def calculate_flash_kb(self, model_type, params):
        """
        Flash storage approximation. For Random Forest, accounts for tree sparsity (gamma).
        """
        if model_type == 'RandomForest':
            nodes_per_tree = int(self.gamma * (2 ** params['max_depth']))
            total_nodes = params['n_estimators'] * nodes_per_tree
            return (total_nodes * 4) / 1024.0 # Weighted average node cost
            
        elif model_type == 'LinearBenchmark':
            total_bytes = (params['n_features'] * self.byte_per_val) + 4 # Weights + bias
            return total_bytes / 1024.0
            
        return 0.0

    def calculate_compute_proxy(self, model_type, params):
        """
        System compute complexity proxy (Multiply-Accumulate operations).
        """
        if model_type == 'RandomForest':
            # Mean traversal cost across forest
            return params['n_estimators'] * params['max_depth']
        elif model_type == 'LinearBenchmark':
            return params['n_features']
        return 0
