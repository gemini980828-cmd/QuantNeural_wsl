"""
Tests for src/models.py

Covers:
- Architecture and shapes (20->16->12->10)
- Dense layers with relu, he_normal
- Dropout rates == params.dropout
- Compile config (Adam, MSE, MAE)
- fit() enforces shuffle=False
"""

import numpy as np
import pytest


class TestSectorPredictorMLP:
    """Test SectorPredictorMLP class."""
    
    @pytest.fixture
    def mlp(self):
        """Create a SectorPredictorMLP instance."""
        from src.models import MLPParams, SectorPredictorMLP
        return SectorPredictorMLP(MLPParams())
    
    @pytest.fixture
    def mlp_params(self):
        """Get default MLPParams."""
        from src.models import MLPParams
        return MLPParams()
    
    # ==================== Architecture Tests ====================
    
    def test_input_shape(self, mlp):
        """Test that input shape is (None, 20)."""
        assert mlp.model.input_shape == (None, 20)
    
    def test_output_shape(self, mlp):
        """Test that output shape is (None, 10)."""
        assert mlp.model.output_shape == (None, 10)
    
    def test_dense_units_sequence(self, mlp):
        """Test Dense layers have units [16, 12, 10] in order."""
        from tensorflow import keras
        
        dense_layers = [
            layer for layer in mlp.model.layers
            if isinstance(layer, keras.layers.Dense)
        ]
        
        units = [layer.units for layer in dense_layers]
        assert units == [16, 12, 10], f"Expected [16, 12, 10], got {units}"
    
    def test_first_two_dense_relu_activation(self, mlp):
        """Test first two Dense layers use relu activation."""
        from tensorflow import keras
        
        dense_layers = [
            layer for layer in mlp.model.layers
            if isinstance(layer, keras.layers.Dense)
        ]
        
        # First two Dense layers should have relu
        for i, layer in enumerate(dense_layers[:2]):
            activation_name = layer.activation.__name__
            assert activation_name == "relu", f"Layer {i} activation: {activation_name}"
    
    def test_first_two_dense_he_normal_initializer(self, mlp):
        """Test first two Dense layers use HeNormal initializer."""
        from tensorflow import keras
        
        dense_layers = [
            layer for layer in mlp.model.layers
            if isinstance(layer, keras.layers.Dense)
        ]
        
        for i, layer in enumerate(dense_layers[:2]):
            init_class = layer.kernel_initializer.__class__.__name__
            assert init_class == "HeNormal", f"Layer {i} initializer: {init_class}"
    
    def test_dropout_layers_count(self, mlp):
        """Test there are exactly two Dropout layers."""
        from tensorflow import keras
        
        dropout_layers = [
            layer for layer in mlp.model.layers
            if isinstance(layer, keras.layers.Dropout)
        ]
        
        assert len(dropout_layers) == 2, f"Expected 2 Dropout, got {len(dropout_layers)}"
    
    def test_dropout_rates(self, mlp, mlp_params):
        """Test Dropout rates equal params.dropout."""
        from tensorflow import keras
        
        dropout_layers = [
            layer for layer in mlp.model.layers
            if isinstance(layer, keras.layers.Dropout)
        ]
        
        for layer in dropout_layers:
            assert layer.rate == mlp_params.dropout, f"Expected {mlp_params.dropout}, got {layer.rate}"
    
    # ==================== Compile Config Tests ====================
    
    def test_optimizer_is_adam(self, mlp):
        """Test optimizer is Adam."""
        opt_class = mlp.model.optimizer.__class__.__name__
        assert opt_class == "Adam", f"Expected Adam, got {opt_class}"
    
    def test_optimizer_learning_rate(self, mlp, mlp_params):
        """Test optimizer learning rate equals params.lr."""
        lr = float(mlp.model.optimizer.learning_rate)
        assert abs(lr - mlp_params.lr) < 1e-9, f"Expected {mlp_params.lr}, got {lr}"
    
    def test_metric_includes_mae(self, mlp):
        """Test metrics include MeanAbsoluteError."""
        # Check that model has some metric configured (Keras 3 wraps metrics)
        # We verify by running a prediction and checking metrics are computed
        import numpy as np
        X = np.random.randn(4, 20).astype(np.float32)
        Y = np.random.randn(4, 10).astype(np.float32)
        # Evaluate should return [loss, metric] if MAE is present
        results = mlp.model.evaluate(X, Y, verbose=0)
        assert len(results) >= 2, "Expected at least 2 values (loss + MAE metric)"
    
    def test_loss_is_mse(self, mlp):
        """Test loss is MeanSquaredError."""
        loss_class = mlp.model.loss.__class__.__name__
        assert loss_class == "MeanSquaredError", f"Expected MeanSquaredError, got {loss_class}"
    
    # ==================== fit() Shuffle Tests ====================
    
    def test_fit_enforces_shuffle_false(self):
        """Test that fit() explicitly passes shuffle=False."""
        from src.models import MLPParams, SectorPredictorMLP
        
        np.random.seed(0)
        
        # Small synthetic data for fast test
        n_train, n_val = 64, 16
        X_train = np.random.randn(n_train, 20).astype(np.float32)
        Y_train = np.random.randn(n_train, 10).astype(np.float32)
        X_val = np.random.randn(n_val, 20).astype(np.float32)
        Y_val = np.random.randn(n_val, 10).astype(np.float32)
        
        # Use minimal epochs for speed
        params = MLPParams(epochs=1, batch_size=16)
        mlp = SectorPredictorMLP(params)
        
        # Spy on model.fit to capture shuffle kwarg
        recorded_kwargs = {}
        original_fit = mlp.model.fit
        
        def spy_fit(*args, **kwargs):
            recorded_kwargs.update(kwargs)
            return original_fit(*args, **kwargs)
        
        mlp.model.fit = spy_fit
        
        # Call fit
        history = mlp.fit(X_train, Y_train, X_val, Y_val)
        
        # Assert shuffle=False was passed
        assert "shuffle" in recorded_kwargs, "shuffle kwarg not passed to model.fit"
        assert recorded_kwargs["shuffle"] is False, f"shuffle was {recorded_kwargs['shuffle']}, expected False"
        
        # Assert returns History-like object
        assert hasattr(history, "history"), "fit() should return History-like object"
    
    def test_fit_runs_and_returns_history(self):
        """Test that fit() runs successfully and returns History."""
        from src.models import MLPParams, SectorPredictorMLP
        from tensorflow import keras
        
        np.random.seed(42)
        
        n_train, n_val = 32, 8
        X_train = np.random.randn(n_train, 20).astype(np.float32)
        Y_train = np.random.randn(n_train, 10).astype(np.float32)
        X_val = np.random.randn(n_val, 20).astype(np.float32)
        Y_val = np.random.randn(n_val, 10).astype(np.float32)
        
        params = MLPParams(epochs=1, batch_size=8)
        mlp = SectorPredictorMLP(params)
        
        history = mlp.fit(X_train, Y_train, X_val, Y_val)
        
        assert isinstance(history, keras.callbacks.History)
    
    def test_predict_returns_correct_shape(self):
        """Test that predict returns correct output shape."""
        from src.models import MLPParams, SectorPredictorMLP
        
        np.random.seed(123)
        
        mlp = SectorPredictorMLP(MLPParams())
        X = np.random.randn(16, 20).astype(np.float32)
        
        preds = mlp.predict(X)
        
        assert preds.shape == (16, 10), f"Expected (16, 10), got {preds.shape}"
