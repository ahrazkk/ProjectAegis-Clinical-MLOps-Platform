import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model')))
from macroscopic_ddi_gnn import MacroscopicDDIGNN

data = torch.load('data/neo4j_gnn_dataset.pt', weights_only=False)
model = MacroscopicDDIGNN(in_channels=data.num_features, hidden_channels=256, out_channels=128, num_layers=3)
model.load_state_dict(torch.load('data/macroscopic_gnn_weights.pth', weights_only=True))
model.eval()

n = 5000
src = torch.randint(0, data.num_nodes, (n,))
dst = torch.randint(0, data.num_nodes, (n,))
edge_label_index = torch.stack([src, dst])

with torch.no_grad():
    out = model(data, edge_label_index)
    probs = torch.sigmoid(out)

print(f'Mean prob: {probs.mean().item():.4f}')
print(f'Min prob: {probs.min().item():.4f}')
print(f'Max prob: {probs.max().item():.4f}')
