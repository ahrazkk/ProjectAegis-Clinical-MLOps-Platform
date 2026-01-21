"""
Risk Scoring Logic with Temperature Scaling Calibration
Implements post-processing to convert model probabilities into clinically actionable risk scores
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union, Tuple


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for Probability Calibration
    
    Applies a learnable temperature parameter to scale logits before softmax,
    improving calibration of neural network outputs.
    
    Reference: MCR III Section 3.1
    """
    
    def __init__(self, initial_temperature: float = 1.5):
        """
        Initialize Temperature Scaling
        
        Args:
            initial_temperature: Initial temperature value (> 0)
        """
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Raw model logits [batch_size, num_classes]
        
        Returns:
            calibrated_probs: Temperature-scaled probabilities
        """
        # Scale logits by temperature
        scaled_logits = logits / self.temperature
        
        # Apply softmax
        calibrated_probs = torch.softmax(scaled_logits, dim=-1)
        
        return calibrated_probs
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Fit temperature parameter on validation set
        
        Args:
            logits: Validation set logits [n_samples, num_classes]
            labels: True labels [n_samples]
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)


class RiskScorer:
    """
    Risk Score Calculator with Severity-Weighted Probabilities
    
    Converts calibrated model probabilities into clinical risk scores
    and categorizes them into actionable risk levels.
    
    Reference: MCR III Section 3.2 & 3.3
    """
    
    # Severity weights as defined in specification
    SEVERITY_WEIGHTS = {
        'minor': 0.2,      # e.g., "Advice"
        'moderate': 0.6,   # e.g., "Effect"
        'major': 0.9       # e.g., "Mechanism-Pharmacokinetic"
    }
    
    # Risk categorization thresholds
    RISK_THRESHOLDS = {
        'low': (0.0, 0.3),
        'moderate': (0.3, 0.7),
        'high': (0.7, 1.0)
    }
    
    def __init__(
        self,
        class_to_severity: Dict[int, str],
        severity_weights: Dict[str, float] = None
    ):
        """
        Initialize Risk Scorer
        
        Args:
            class_to_severity: Mapping from class index to severity level
                               e.g., {0: 'none', 1: 'minor', 2: 'moderate', 3: 'major'}
            severity_weights: Custom severity weights (optional)
        """
        self.class_to_severity = class_to_severity
        self.severity_weights = severity_weights or self.SEVERITY_WEIGHTS
    
    def calculate_risk_score(
        self,
        probabilities: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Calculate risk score using weighted sum formula
        
        Formula: R = Σ(P_calibrated_i × W_i)
        
        Args:
            probabilities: Calibrated class probabilities [batch_size, num_classes]
        
        Returns:
            risk_scores: Scalar risk scores [batch_size]
        """
        is_torch = isinstance(probabilities, torch.Tensor)
        
        if is_torch:
            device = probabilities.device
            probabilities_np = probabilities.detach().cpu().numpy()
        else:
            probabilities_np = probabilities
        
        # Create weight vector
        num_classes = probabilities_np.shape[-1]
        weights = np.zeros(num_classes)
        
        for class_idx, severity in self.class_to_severity.items():
            if severity in self.severity_weights:
                weights[class_idx] = self.severity_weights[severity]
        
        # Calculate weighted sum: R = Σ(P_i × W_i)
        risk_scores = np.dot(probabilities_np, weights)
        
        if is_torch:
            risk_scores = torch.tensor(risk_scores, device=device)
        
        return risk_scores
    
    def categorize_risk(
        self,
        risk_scores: Union[torch.Tensor, np.ndarray, float]
    ) -> Union[str, list]:
        """
        Categorize risk scores into Low/Moderate/High
        
        Thresholds:
        - Low Risk: R < 0.3
        - Moderate Risk: 0.3 ≤ R < 0.7
        - High Risk: R ≥ 0.7
        
        Args:
            risk_scores: Risk scores (scalar or array)
        
        Returns:
            categories: Risk category labels
        """
        # Convert to numpy for easier processing
        if isinstance(risk_scores, torch.Tensor):
            risk_scores = risk_scores.detach().cpu().numpy()
        
        is_scalar = np.isscalar(risk_scores)
        if is_scalar:
            risk_scores = np.array([risk_scores])
        
        categories = []
        for score in risk_scores:
            if score < self.RISK_THRESHOLDS['low'][1]:
                categories.append('low')
            elif score < self.RISK_THRESHOLDS['moderate'][1]:
                categories.append('moderate')
            else:
                categories.append('high')
        
        return categories[0] if is_scalar else categories
    
    def get_risk_assessment(
        self,
        probabilities: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[str, list]]:
        """
        Complete risk assessment: calculate scores and categorize
        
        Args:
            probabilities: Calibrated class probabilities
        
        Returns:
            risk_scores: Numerical risk scores
            risk_categories: Categorical risk levels
        """
        risk_scores = self.calculate_risk_score(probabilities)
        risk_categories = self.categorize_risk(risk_scores)
        
        return risk_scores, risk_categories
