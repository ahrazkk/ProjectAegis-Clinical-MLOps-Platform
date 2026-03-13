import os
import torch
from neo4j import GraphDatabase

try:
    from torch_geometric.data import Data
    import torch_geometric.transforms as T
except ImportError:
    print("FATAL: torch_geometric must be installed. Try 'pip install torch_geometric'")
    exit(1)
    
# Import the brand new GNN model architecture we wrote
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model')))
from macroscopic_ddi_gnn import MacroscopicDDIGNN, train_link_predictor

def main():
    print("==============================================")
    print("   TRAINING MACROSCOPIC GNN ON NEO4J DATA")
    print("==============================================")
    
    data_path = os.path.join(os.path.dirname(__file__), "data", "neo4j_gnn_dataset.pt")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find '{data_path}'")
        print("Please run 'extract_graph_dataset.py' to generate the tensors first.")
        return
        
    print("Loading PyG Tensor Dataset...")
    data = torch.load(data_path, weights_only=False)
    print(f"Dataset Details:")
    print(f" - Nodes (Drugs): {data.num_nodes}")
    print(f" - Edges (Known Interactions): {data.num_edges//2} (undirected)")
    print(f" - Features per Drug (Chemistry+Biology): {data.num_features}")
    
    # Initialize the specific model
    print("\nInitializing GraphSAGE Model...")
    model = MacroscopicDDIGNN(in_channels=data.num_features, hidden_channels=256, out_channels=128, num_layers=3)
    
    # Run the training loop!
    # Because Link Prediction is difficult, we run 150 epochs.
    trained_model = train_link_predictor(model, data, epochs=150, lr=0.005)
    
    if trained_model:
        save_file = os.path.join(os.path.dirname(__file__), "data", "macroscopic_gnn_weights.pth")
        torch.save(trained_model.state_dict(), save_file)
        print(f"\n[SUCCESS] Model weights saved to: {save_file}")
        
if __name__ == '__main__':
    main()
