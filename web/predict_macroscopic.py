import os
import sys
import torch
import pandas as pd
from difflib import get_close_matches

try:
    from torch_geometric.data import Data
except ImportError:
    print("FATAL: torch_geometric must be installed.")
    exit(1)

# Import the Macro GNN mapping definition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model')))
from macroscopic_ddi_gnn import MacroscopicDDIGNN

class MacroscopicPredictor:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.dataset_path = os.path.join(self.data_dir, "neo4j_gnn_dataset.pt")
        self.mapping_path = os.path.join(self.data_dir, "node_mapping.csv")
        self.weights_path = os.path.join(self.data_dir, "macroscopic_gnn_weights.pth")
        
        self.model = None
        self.graph_data = None
        self.mapping_df = None
        self.name_to_idx = {}
        
        self._load_assets()
        
    def _load_assets(self):
        print("Loading Knowledge Graph and Tensor Mappings...")
        if not os.path.exists(self.dataset_path) or \
           not os.path.exists(self.mapping_path) or \
           not os.path.exists(self.weights_path):
            raise FileNotFoundError("Missing highly dense graph tensors or trained weights!")
            
        # 1. Load Graph Tensors
        self.graph_data = torch.load(self.dataset_path, weights_only=False)
        
        # 2. Load the Node Name Mapping Dictionary
        self.mapping_df = pd.read_csv(self.mapping_path)
        
        # Build dictionary mapping lowercased names to PyG node index
        for _, row in self.mapping_df.iterrows():
            if pd.notna(row['name']):
                # Standardize inputs to lowercase
                clean_name = str(row['name']).strip().lower()
                self.name_to_idx[clean_name] = row['pyg_id']
                
        # 3. Load the Model Architecture
        # NOTE: Must match exactly the hyperparams from train_macroscopic_model.py
        in_channels = self.graph_data.num_features
        self.model = MacroscopicDDIGNN(in_channels=in_channels, hidden_channels=256, out_channels=128, num_layers=3)
        self.model.load_state_dict(torch.load(self.weights_path, weights_only=True))
        self.model.eval() # Set to evaluation mode!
        print("[System] Macroscopic GraphSAGE Model Loaded & Ready.")

    def find_drug(self, search_name):
        search_name = search_name.strip().lower()
        if search_name in self.name_to_idx:
            return search_name, self.name_to_idx[search_name]
            
        # Try to find a close match using difflib if spelled wrong
        close_matches = get_close_matches(search_name, self.name_to_idx.keys(), n=1, cutoff=0.7)
        if close_matches:
            match = close_matches[0]
            print(f"Warning: Could not find '{search_name}', auto-correcting to '{match}'")
            return match, self.name_to_idx[match]
            
        return None, None

    def predict_interaction(self, drug1_name, drug2_name):
        d1_clean, idx1 = self.find_drug(drug1_name)
        d2_clean, idx2 = self.find_drug(drug2_name)
        
        if idx1 is None:
            return f"Error: Drug '{drug1_name}' is not in the Neo4j Graph Network."
        if idx2 is None:
            return f"Error: Drug '{drug2_name}' is not in the Neo4j Graph Network."
            
        if idx1 == idx2:
            return "Error: Please input two different drugs."
            
        print(f"\n--- Running Inference ---")
        print(f"Target 1: {d1_clean.title()} (Node ID: {idx1})")
        print(f"Target 2: {d2_clean.title()} (Node ID: {idx2})")
        
        # Build the exact edge label index tensor to query the GNN
        edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long)
        
        # Execute the forward pass!
        with torch.no_grad():
            # Pass the entire global graph + the two nodes we want to test
            out = self.model(self.graph_data, edge_label_index)
            raw_logit = out.item()
            # Apply Sigmoid to convert the raw logit output to a percentage probability [0.0 - 1.0]
            probability = torch.sigmoid(out).item()
            
        percentage = probability * 100
        
        print("\n=== Result ===")
        print(f"Raw Model Logit: {raw_logit:.4f}")
        print(f"Adverse Interaction Probability: {percentage:.2f}%")
        
        if percentage > 85.0:
            print("ALERT: SEVERE DANGER OF ADVERSE INTERACTION.")
        elif percentage > 60.0:
            print("WARNING: MODERATE RISK OF INTERACTION.")
        else:
            print("SAFE: MINIMAL PREDICTED GRAPH INTERACTION.")
            
        return percentage

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_macroscopic.py \"Drug Name 1\" \"Drug Name 2\"")
        print("Example: python predict_macroscopic.py \"Aspirin\" \"Warfarin\"")
        return
        
    predictor = MacroscopicPredictor()
    predictor.predict_interaction(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
