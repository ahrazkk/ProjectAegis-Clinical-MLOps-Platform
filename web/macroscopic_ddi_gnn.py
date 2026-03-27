import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool
except ImportError:
    pass # Managed in the exception block below when file is run

class MacroscopicDDIGNN(nn.Module):
    """
    Macroscopic Graph Neural Network for DDI Link Prediction.
    
    Why this is better than the Molecular (Microscopic) GNN:
    1. OLD WAY: Looks at a single drug and tries to predict chemistry from atoms. 
       Zero knowledge of how the human body works or how other drugs behave.
    2. NEW WAY: Looks at the ENTIRE database as a giant web (Drug Network).
       Uses GraphSAGE to pass biological "messages" between drugs. 
       If Drug A interacts with Drug B, and Drug C is identical to Drug B, 
       the model will intuitively predict an A->C interaction based on graph proximity.
    """
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, num_layers=3):
        super(MacroscopicDDIGNN, self).__init__()
        
        # We use GraphSAGE. SAGE is vastly superior to pure GCN for highly connected, 
        # heterogeneous graphs because it samples local neighborhoods effectively,
        # preventing the nodes from becoming too "smoothed out" (oversmoothing).
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        # Dropout for regularization during link prediction
        self.dropout = nn.Dropout(0.3)

    def encode(self, x, edge_index):
        """
        Embeds the node features (SMILES Chemistry + Biological Classes) 
        into the graph space by passing messages along the known interaction edges.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1: # No ReLU on output layer
                x = F.relu(x)
                x = self.dropout(x)
        return x

    def decode(self, z, edge_label_index):
        """
        Takes the embedded nodes and predicts if an edge exists between them.
        Uses a dot product between the two node vectors to calculate interaction probability.
        """
        # z: the embedded vectors for all drugs (shape: [num_drugs, out_channels])
        # edge_label_index: the specific pairs of drugs we want to test (shape: [2, num_test_pairs])
        
        src = z[edge_label_index[0]] # Get embeddings for drug A
        dst = z[edge_label_index[1]] # Get embeddings for drug B
        
        # Dot product prediction: 
        # If the vectors point in the same direction in the latent space, 
        # they are highly likely to interact.
        return (src * dst).sum(dim=-1)

    def forward(self, data, edge_label_index):
        """Standard PyTorch forward pass."""
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, edge_label_index)

class FocalLoss(nn.Module):
    """
    Focal Loss to heavily penalize false positives. 
    It forces the network to focus on harder examples rather than defaulting to 'safe' guesses.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Prevents nans
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# -------------- TRAINING PIPELINE --------------
def train_link_predictor(model, data, epochs=100, lr=0.01):
    """
    Trains the GNN using Focal Loss to balance out over-predicting false positives.
    """
    try:
        import torch_geometric.transforms as T
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("Please install torch_geometric and scikit-learn for training.")
        return

    transform = T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Replaced BCEWithLogitsLoss with FocalLoss to fix Alert Fatigue (False Positives)
    criterion = FocalLoss(alpha=0.8, gamma=2.0)

    print(f"Beginning Training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on the graph
        out = model(train_data, train_data.edge_label_index)
        loss = criterion(out, train_data.edge_label.float())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Validate
                val_out = model(val_data, val_data.edge_label_index)
                val_loss = criterion(val_out, val_data.edge_label.float())
                
                # Calculate metrics
                val_preds = torch.sigmoid(val_out).cpu().numpy()
                val_labels = val_data.edge_label.cpu().numpy()
                auc = roc_auc_score(val_labels, val_preds)
                
                print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val AUC: {auc:.4f}")

    # Final Test
    model.eval()
    with torch.no_grad():
        test_out = model(test_data, test_data.edge_label_index)
        test_preds = torch.sigmoid(test_out).cpu().numpy()
        test_labels = test_data.edge_label.cpu().numpy()
        test_auc = roc_auc_score(test_labels, test_preds)
        test_ap = average_precision_score(test_labels, test_preds)
        
    print("\n--- Training Complete ---")
    print(f"Final Test AUC: {test_auc:.4f} (Accuracy of detecting true vs false interactions)")
    print(f"Final Test AP:  {test_ap:.4f} (Precision-Recall score)")
    
    return model

if __name__ == "__main__":
    # Test script entry point
    print("Macroscopic GNN Architecture Initialized. Run via main pipeline to train.")
