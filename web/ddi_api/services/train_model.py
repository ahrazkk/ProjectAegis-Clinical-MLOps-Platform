"""
GNN Model Training Pipeline for Drug-Drug Interaction Prediction
Trains a Graph Neural Network on molecular structures and knowledge graph embeddings
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# Django setup
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
django.setup()

from ddi_api.services.knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
MODEL_DIR = Path(__file__).parent.parent.parent / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Molecular Feature Extraction
# ============================================================================

def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert SMILES string to Morgan fingerprint"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception as e:
        logger.warning(f"Failed to compute fingerprint for {smiles}: {e}")
        return np.zeros(n_bits)


def smiles_to_descriptors(smiles: str) -> np.ndarray:
    """Extract molecular descriptors from SMILES"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(10)
        
        descriptors = [
            Descriptors.MolWt(mol),           # Molecular weight
            Descriptors.MolLogP(mol),         # LogP (lipophilicity)
            Descriptors.TPSA(mol),            # Topological polar surface area
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Lipinski.FractionCSP3(mol),       # Fraction of sp3 carbons
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.HeavyAtomCount(mol),
        ]
        return np.array(descriptors, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to compute descriptors for {smiles}: {e}")
        return np.zeros(10)


# ============================================================================
# Dataset
# ============================================================================

class DDIDataset(Dataset):
    """Dataset for drug-drug interaction prediction"""
    
    SEVERITY_MAP = {'severe': 2, 'moderate': 1, 'minor': 0, 'none': -1}
    
    def __init__(self, drug_pairs: List[Dict], drug_features: Dict[str, np.ndarray]):
        """
        Args:
            drug_pairs: List of {'drug1': id, 'drug2': id, 'severity': str, ...}
            drug_features: Dict mapping drug_id -> feature vector
        """
        self.pairs = drug_pairs
        self.drug_features = drug_features
        self.feature_dim = next(iter(drug_features.values())).shape[0] if drug_features else 2048
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Get drug features
        drug1_id = pair['drug1']
        drug2_id = pair['drug2']
        
        feat1 = self.drug_features.get(drug1_id, np.zeros(self.feature_dim))
        feat2 = self.drug_features.get(drug2_id, np.zeros(self.feature_dim))
        
        # Combine features (concatenation + element-wise operations)
        combined = np.concatenate([
            feat1, 
            feat2, 
            feat1 * feat2,  # Hadamard product
            np.abs(feat1 - feat2)  # Absolute difference
        ])
        
        # Get label
        severity = pair.get('severity', 'none')
        label = self.SEVERITY_MAP.get(severity, -1)
        
        # For binary classification: has_interaction
        has_interaction = 1 if label >= 0 else 0
        
        return {
            'features': torch.tensor(combined, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'has_interaction': torch.tensor(has_interaction, dtype=torch.float32),
            'drug1': drug1_id,
            'drug2': drug2_id
        }


# ============================================================================
# Neural Network Models
# ============================================================================

class MolecularEncoder(nn.Module):
    """Encodes molecular fingerprints into embeddings"""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)


class DDIClassifier(nn.Module):
    """
    Drug-Drug Interaction Classifier
    Takes concatenated drug pair features and predicts interaction severity
    """
    
    def __init__(self, input_dim: int = 2048 * 4, hidden_dim: int = 512, 
                 num_classes: int = 3, temperature: float = 1.5):
        super().__init__()
        
        self.temperature = temperature
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Binary classifier: does interaction exist?
        self.interaction_head = nn.Linear(hidden_dim // 4, 1)
        
        # Severity classifier: minor/moderate/severe
        self.severity_head = nn.Linear(hidden_dim // 4, num_classes)
        
        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_dim // 4, 1)
    
    def forward(self, x):
        features = self.feature_net(x)
        
        # Interaction probability
        interaction_logit = self.interaction_head(features)
        interaction_prob = torch.sigmoid(interaction_logit)
        
        # Severity logits (with temperature scaling for calibration)
        severity_logits = self.severity_head(features) / self.temperature
        
        # Confidence score
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'interaction_prob': interaction_prob.squeeze(-1),
            'severity_logits': severity_logits,
            'confidence': confidence.squeeze(-1),
            'features': features
        }
    
    def predict(self, x):
        """Get predictions with severity labels"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            severity_probs = F.softmax(outputs['severity_logits'], dim=-1)
            severity_idx = torch.argmax(severity_probs, dim=-1)
            
            severity_labels = ['minor', 'moderate', 'severe']
            
            return {
                'has_interaction': (outputs['interaction_prob'] > 0.5).cpu().numpy(),
                'interaction_probability': outputs['interaction_prob'].cpu().numpy(),
                'severity': [severity_labels[i] for i in severity_idx.cpu().numpy()],
                'severity_probabilities': severity_probs.cpu().numpy(),
                'confidence': outputs['confidence'].cpu().numpy()
            }


# ============================================================================
# Training Pipeline
# ============================================================================

class DDITrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    def train_epoch(self, dataloader: DataLoader, optimizer, criterion_interaction, criterion_severity):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            has_interaction = batch['has_interaction'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(features)
            
            # Binary interaction loss
            loss_interaction = F.binary_cross_entropy(
                outputs['interaction_prob'], 
                has_interaction
            )
            
            # Severity loss (only for pairs with known interactions)
            mask = labels >= 0
            if mask.sum() > 0:
                loss_severity = criterion_severity(
                    outputs['severity_logits'][mask],
                    labels[mask]
                )
            else:
                loss_severity = 0
            
            # Combined loss
            loss = loss_interaction + 0.5 * loss_severity
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        all_probs = []
        all_labels = []
        all_severity_probs = []
        all_severity_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                has_interaction = batch['has_interaction'].numpy()
                labels = batch['label'].numpy()
                
                outputs = self.model(features)
                
                all_probs.extend(outputs['interaction_prob'].cpu().numpy())
                all_labels.extend(has_interaction)
                
                mask = labels >= 0
                if mask.sum() > 0:
                    severity_probs = F.softmax(outputs['severity_logits'], dim=-1)
                    all_severity_probs.extend(severity_probs[mask].cpu().numpy())
                    all_severity_labels.extend(labels[mask])
        
        # Metrics
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary', zero_division=0
        )
        
        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, lr: float = 1e-3, patience: int = 10):
        """Full training loop with early stopping"""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        criterion_interaction = nn.BCELoss()
        criterion_severity = nn.CrossEntropyLoss()
        
        best_auc = 0
        no_improve = 0
        
        logger.info(f"Training on {self.device} for {epochs} epochs")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(
                train_loader, optimizer, criterion_interaction, criterion_severity
            )
            
            val_metrics = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_auc'].append(val_metrics['auc'])
            
            scheduler.step(val_metrics['auc'])
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {train_loss:.4f} - "
                f"Val AUC: {val_metrics['auc']:.4f} - "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Early stopping
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                no_improve = 0
                self.save_model('best_model.pt')
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"Training complete. Best AUC: {best_auc:.4f}")
        return self.history
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        path = MODEL_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        path = MODEL_DIR / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        logger.info(f"Model loaded from {path}")


# ============================================================================
# Main Training Script
# ============================================================================

def prepare_training_data():
    """Prepare training data from Neo4j knowledge graph"""
    kg = KnowledgeGraphService
    
    if not kg.is_connected():
        logger.warning("Neo4j not connected, using sample data")
        return prepare_sample_training_data()
    
    # Get all drugs with SMILES
    drugs_query = """
    MATCH (d:Drug) 
    WHERE d.smiles IS NOT NULL
    RETURN d.drugbank_id as id, d.name as name, d.smiles as smiles
    """
    drugs = kg.run_query(drugs_query)
    
    # Get all interactions
    interactions_query = """
    MATCH (d1:Drug)-[i:INTERACTS_WITH]->(d2:Drug)
    RETURN d1.drugbank_id as drug1, d2.drugbank_id as drug2, i.severity as severity
    """
    interactions = kg.run_query(interactions_query)
    
    # Compute features for each drug
    drug_features = {}
    for drug in drugs:
        if drug['smiles']:
            fp = smiles_to_fingerprint(drug['smiles'])
            drug_features[drug['id']] = fp
    
    # Generate negative samples (non-interacting pairs)
    drug_ids = list(drug_features.keys())
    interacting_pairs = set((i['drug1'], i['drug2']) for i in interactions)
    interacting_pairs.update((i['drug2'], i['drug1']) for i in interactions)
    
    negative_samples = []
    for i, d1 in enumerate(drug_ids):
        for d2 in drug_ids[i+1:]:
            if (d1, d2) not in interacting_pairs:
                negative_samples.append({
                    'drug1': d1,
                    'drug2': d2,
                    'severity': 'none'
                })
    
    # Balance dataset (2:1 negative to positive ratio)
    n_positives = len(interactions)
    negative_samples = negative_samples[:n_positives * 2]
    
    all_pairs = interactions + negative_samples
    
    logger.info(f"Prepared {len(drug_features)} drugs, {len(interactions)} positive, {len(negative_samples)} negative pairs")
    
    return all_pairs, drug_features


def prepare_sample_training_data():
    """Prepare sample training data when Neo4j is unavailable"""
    from ddi_api.services.data_ingestion import DrugDataIngestion
    
    ingestion = DrugDataIngestion()
    
    # Get drug features
    drug_features = {}
    for drug in ingestion.SAMPLE_DRUGS:
        if drug.get('smiles'):
            fp = smiles_to_fingerprint(drug['smiles'])
            drug_features[drug['id']] = fp
    
    # Get interactions
    interactions = []
    for inter in ingestion.SAMPLE_INTERACTIONS:
        drug1 = inter.get('drug1')
        drug2 = inter.get('drug2')
        if drug1 and drug2:
            interactions.append({
                'drug1': drug1,
                'drug2': drug2,
                'severity': inter['severity']
            })
    
    # Generate negative samples (non-interacting pairs)
    drug_ids = list(drug_features.keys())
    interacting_pairs = set((i['drug1'], i['drug2']) for i in interactions)
    interacting_pairs.update((i['drug2'], i['drug1']) for i in interactions)
    
    negative_samples = []
    for i, d1 in enumerate(drug_ids):
        for d2 in drug_ids[i+1:]:
            if (d1, d2) not in interacting_pairs:
                negative_samples.append({
                    'drug1': d1,
                    'drug2': d2,
                    'severity': 'none'
                })
    
    # Balance dataset
    n_positives = len(interactions)
    negative_samples = negative_samples[:n_positives * 2]  # 2:1 ratio
    
    all_pairs = interactions + negative_samples
    
    logger.info(f"Sample data: {len(drug_features)} drugs, {len(interactions)} positive, {len(negative_samples)} negative")
    
    return all_pairs, drug_features


def run_training(epochs: int = 50, batch_size: int = 32):
    """Run the full training pipeline"""
    logger.info("=" * 50)
    logger.info("DDI Model Training Pipeline")
    logger.info("=" * 50)
    
    # Prepare data
    pairs, drug_features = prepare_training_data()
    
    if len(pairs) < 10:
        logger.error("Not enough training data!")
        return None
    
    # Split data
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = DDIDataset(train_pairs, drug_features)
    val_dataset = DDIDataset(val_pairs, drug_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    input_dim = train_dataset.feature_dim * 4  # Concatenated features
    model = DDIClassifier(input_dim=input_dim)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = DDITrainer(model)
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Save final model
    trainer.save_model('ddi_model_final.pt')
    
    # Save training metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'num_drugs': len(drug_features),
        'num_pairs': len(pairs),
        'epochs': epochs,
        'best_auc': max(history['val_auc']),
        'feature_dim': train_dataset.feature_dim
    }
    
    with open(MODEL_DIR / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training complete!")
    return trainer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DDI Prediction Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    run_training(epochs=args.epochs, batch_size=args.batch_size)
