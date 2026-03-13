"""
Platt scaling calibration for GNN model.

Learns parameters a, b such that P(DDI) = sigmoid(a * logit + b)
on the validation set. This fixes overconfident predictions.
"""
import json
import sys
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gnn_featurizer import MolecularGraphFeaturizer, ATOM_FEATURE_DIM, EDGE_FEATURE_DIM
from model.gnn_model import DDIGraphModel
from model.gnn_dataset import DDIGraphDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / 'web' / 'models' / 'gnn'
DATA_DIR = PROJECT_ROOT / 'web' / 'data' / 'gnn_training'


def collect_val_logits():
    """Run model on validation set and collect (logit, label) pairs."""
    with open(MODEL_DIR / 'training_results.json') as f:
        cfg = json.load(f)['config']

    model = DDIGraphModel(
        atom_feature_dim=ATOM_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=cfg['hidden_dim'],
        num_gnn_layers=cfg['num_gnn_layers'],
        dropout_rate=cfg['dropout_rate'],
        num_relation_classes=1,
        use_jumping_knowledge=cfg['use_jumping_knowledge'],
    )
    checkpoint = torch.load(MODEL_DIR / 'gnn_best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    featurizer = MolecularGraphFeaturizer(max_atoms=cfg['max_atoms'])
    val_dataset = DDIGraphDataset.from_json(
        str(DATA_DIR / 'val.json'), featurizer, use_binary_labels=True
    )

    logits_list = []
    labels_list = []

    loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch['drug1_node_features'], batch['drug1_adjacency'],
                batch['drug1_edge_features'], batch['drug1_node_mask'],
                batch['drug2_node_features'], batch['drug2_adjacency'],
                batch['drug2_edge_features'], batch['drug2_node_mask'],
            )
            logits_list.append(out.squeeze(-1))
            labels_list.append(batch['relation_label'].float())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    return logits, labels


def fit_platt_scaling(logits, labels, lr=0.01, max_iter=1000):
    """Fit Platt scaling parameters a, b via log-loss on val set."""
    a = nn.Parameter(torch.tensor(1.0))
    b = nn.Parameter(torch.tensor(0.0))
    optimizer = optim.LBFGS([a, b], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(a * logits + b, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    final_loss = criterion(a * logits.detach() + b, labels).item()
    logger.info(f"Platt scaling: a={a.item():.4f}, b={b.item():.4f}, loss={final_loss:.4f}")
    return a.item(), b.item()


def evaluate_calibration(logits, labels, a, b):
    """Show calibration results."""
    calibrated = torch.sigmoid(a * logits + b)
    uncalibrated = torch.sigmoid(logits)

    # Binary accuracy at 0.5 threshold
    for name, probs in [("Uncalibrated", uncalibrated), ("Calibrated", calibrated)]:
        preds = (probs >= 0.5).float()
        acc = (preds == labels).float().mean().item()
        pos_mean = probs[labels == 1].mean().item()
        neg_mean = probs[labels == 0].mean().item()
        logger.info(f"{name}: Acc={acc:.3f}, Mean(pos)={pos_mean:.3f}, Mean(neg)={neg_mean:.3f}")

    return calibrated


def main():
    logger.info("Collecting validation logits...")
    logits, labels = collect_val_logits()
    logger.info(f"Val samples: {len(logits)} (pos={labels.sum().int()}, neg={(1-labels).sum().int()})")
    logger.info(f"Raw logit range: [{logits.min():.2f}, {logits.max():.2f}]")

    logger.info("\nFitting Platt scaling...")
    a, b = fit_platt_scaling(logits, labels)

    logger.info("\nEvaluating calibration:")
    evaluate_calibration(logits, labels, a, b)

    # Save calibration parameters into the checkpoint
    checkpoint = torch.load(MODEL_DIR / 'gnn_best_model.pt', map_location='cpu', weights_only=False)
    checkpoint['platt_a'] = a
    checkpoint['platt_b'] = b
    torch.save(checkpoint, MODEL_DIR / 'gnn_best_model.pt')
    logger.info(f"\nSaved Platt parameters to checkpoint: a={a:.4f}, b={b:.4f}")

    # Also save to a simple JSON for the predictor
    cal = {'platt_a': a, 'platt_b': b}
    with open(MODEL_DIR / 'calibration.json', 'w') as f:
        json.dump(cal, f, indent=2)
    logger.info(f"Saved calibration.json")

    # Test on known pairs
    logger.info("\n--- Testing on known drug pairs ---")
    with open(PROJECT_ROOT / 'web' / 'data' / 'drug_db.json') as f:
        drugs = {d['name'].lower(): d['smiles'] for d in json.load(f)}

    with open(MODEL_DIR / 'training_results.json') as f:
        cfg = json.load(f)['config']

    model = DDIGraphModel(
        atom_feature_dim=ATOM_FEATURE_DIM, edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=cfg['hidden_dim'], num_gnn_layers=cfg['num_gnn_layers'],
        dropout_rate=cfg['dropout_rate'], num_relation_classes=1,
        use_jumping_knowledge=cfg['use_jumping_knowledge'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    featurizer = MolecularGraphFeaturizer(max_atoms=cfg['max_atoms'])

    test_pairs = [
        ('warfarin', 'aspirin', 'DDI'),
        ('simvastatin', 'amiodarone', 'DDI'),
        ('clopidogrel', 'omeprazole', 'DDI'),
        ('fluoxetine', 'tramadol', 'DDI'),
        ('metformin', 'omeprazole', 'None'),
        ('ibuprofen', 'acetaminophen', 'None'),
        ('lisinopril', 'metformin', 'None'),
        ('amoxicillin', 'metformin', 'None'),
    ]

    correct = total = 0
    for d1, d2, expected in test_pairs:
        s1, s2 = drugs.get(d1), drugs.get(d2)
        if not s1 or not s2:
            continue
        p = featurizer.smiles_pair_to_graphs(s1, s2)
        if p is None:
            continue
        with torch.no_grad():
            logit = model(
                p['drug1_node_features'].unsqueeze(0), p['drug1_adjacency'].unsqueeze(0),
                p['drug1_edge_features'].unsqueeze(0), p['drug1_node_mask'].unsqueeze(0),
                p['drug2_node_features'].unsqueeze(0), p['drug2_adjacency'].unsqueeze(0),
                p['drug2_edge_features'].unsqueeze(0), p['drug2_node_mask'].unsqueeze(0),
            ).squeeze().item()

        raw_prob = 1 / (1 + np.exp(-logit))
        cal_prob = 1 / (1 + np.exp(-(a * logit + b)))
        pred = 'DDI' if cal_prob >= 0.5 else 'None'
        ok = pred == expected
        total += 1
        if ok:
            correct += 1
        mark = 'OK' if ok else 'X'
        logger.info(f"  {d1+' + '+d2:<30} exp={expected:<5} raw={raw_prob:.3f} cal={cal_prob:.3f} {mark}")

    logger.info(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")


if __name__ == '__main__':
    main()
