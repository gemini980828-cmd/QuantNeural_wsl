"""
Models module for QUANT-NEURAL.

Provides:
- MLPParams: Configuration for MLP training.
- SectorPredictorMLP: Keras Functional API MLP for sector prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass(frozen=True)
class MLPParams:
    """Parameters for MLP model training.
    
    Attributes
    ----------
    lr : float
        Learning rate for Adam optimizer.
    dropout : float
        Dropout rate for regularization.
    patience_es : int
        Patience for EarlyStopping callback.
    patience_rlr : int
        Patience for ReduceLROnPlateau callback.
    min_lr : float
        Minimum learning rate for ReduceLROnPlateau.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs.
    """
    lr: float = 1e-3
    dropout: float = 0.3
    patience_es: int = 10
    patience_rlr: int = 5
    min_lr: float = 1e-5
    batch_size: int = 32
    epochs: int = 200


class SectorPredictorMLP:
    """
    MLP model for sector prediction.
    
    Architecture (Keras Functional API):
    - Input: (None, 20)
    - Dense(16, relu, he_normal) + Dropout(0.3)
    - Dense(12, relu, he_normal) + Dropout(0.3)
    - Dense(10, linear)
    
    Parameters
    ----------
    params : MLPParams
        Training configuration.
    
    Attributes
    ----------
    model : keras.Model
        Compiled Keras model.
    """
    
    def __init__(self, params: MLPParams):
        self.params = params
        self.model = self._build()
    
    def _build(self) -> keras.Model:
        """Build and compile the MLP model."""
        inp = keras.Input(shape=(20,), name="x")
        
        x = keras.layers.Dense(
            16, activation="relu", kernel_initializer="he_normal"
        )(inp)
        x = keras.layers.Dropout(self.params.dropout)(x)
        
        x = keras.layers.Dense(
            12, activation="relu", kernel_initializer="he_normal"
        )(x)
        x = keras.layers.Dropout(self.params.dropout)(x)
        
        out = keras.layers.Dense(10, activation=None, name="yhat")(x)
        
        model = keras.Model(inputs=inp, outputs=out, name="SectorPredictorMLP")
        
        opt = keras.optimizers.Adam(learning_rate=self.params.lr)
        model.compile(
            optimizer=opt,
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_train, 20).
        Y_train : np.ndarray
            Training targets, shape (n_train, 10).
        X_val : np.ndarray
            Validation features, shape (n_val, 20).
        Y_val : np.ndarray
            Validation targets, shape (n_val, 10).
        
        Returns
        -------
        keras.callbacks.History
            Training history.
        
        Notes
        -----
        - CRITICAL: shuffle=False is enforced for time-series data.
        - Callbacks: EarlyStopping + ReduceLROnPlateau.
        """
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.params.patience_es,
            restore_best_weights=True,
        )
        rlr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.params.patience_rlr,
            min_lr=self.params.min_lr,
        )
        
        return self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            shuffle=False,          # CRITICAL: no shuffle for time-series
            callbacks=[es, rlr],
            verbose=0,              # keep tests quiet
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, 20).
        
        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples, 10).
        """
        return self.model.predict(X, verbose=0)
