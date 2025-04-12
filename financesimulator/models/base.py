"""Base model interface for simulation models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union


class BaseModel(ABC):
    """Base class for all simulation models."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model with parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        self.params = params
        self._validate_params()
    
    @abstractmethod
    def _validate_params(self) -> None:
        """
        Validate that all required parameters are present and valid.
        
        Raises:
            ValueError: If any parameters are invalid or missing
        """
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseModel':
        """
        Fit the model to historical data.
        
        Args:
            data: Historical data to fit the model
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def simulate(
        self, 
        starting_price: float, 
        horizon: int, 
        paths: int
    ) -> np.ndarray:
        """
        Generate price paths based on the model.
        
        Args:
            starting_price: Initial price
            horizon: Number of steps to simulate
            paths: Number of paths to generate
            
        Returns:
            Array of shape (paths, horizon+1) containing the simulated price paths
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.params.copy()


class ModelFactory:
    """Factory class for creating models."""
    
    _registered_models = {}
    
    @classmethod
    def register_model(cls, model_name: str, model_class: type):
        """
        Register a model class.
        
        Args:
            model_name: Name of the model
            model_class: Model class
        """
        cls._registered_models[model_name.lower()] = model_class
    
    @classmethod
    def create_model(cls, model_name: str, params: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_name is not registered
        """
        model_name = model_name.lower()
        if model_name not in cls._registered_models:
            raise ValueError(f"Model '{model_name}' not registered. Available models: {list(cls._registered_models.keys())}")
        
        model_class = cls._registered_models[model_name]
        return model_class(params)