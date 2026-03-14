with open("ddi_api/services/gnn_predictor.py", "r", encoding="utf-8") as f:
    text = f.read()

import re

new_func = """    def _load_macroscopic_gnn(self):
        \"\"\"Load the V2 Macroscopic GraphSAGE Model and Tensor Space.\"\"\"
        import sys
        import pandas as pd
        model_src = Path(__file__).parent.parent.parent.parent / 'src' / 'model'
        if str(model_src) not in sys.path:
            sys.path.insert(0, str(model_src))
            
        web_dir = Path(__file__).parent.parent.parent
        if str(web_dir) not in sys.path:
            sys.path.insert(0, str(web_dir))
            
        data_dir = Path(__file__).parent.parent.parent / 'data'
        
        try:
            from macroscopic_ddi_gnn import MacroscopicDDIGNN
            
            dataset_path = data_dir / "neo4j_gnn_dataset.pt"
            mapping_path = data_dir / "node_mapping.csv"
            weights_path = data_dir / "macroscopic_gnn_weights.pth"

            if not dataset_path.exists() or not mapping_path.exists() or not weights_path.exists():
                logger.warning("Macroscopic GNN assets missing. Ensure extract_graph_dataset.py and train_macroscopic_model.py have been executed.")
                return

            self.macro_graph_data = torch.load(dataset_path, weights_only=False)
            mapping_df = pd.read_csv(mapping_path)

            for _, row in mapping_df.iterrows():
                if pd.notna(row['name']):
                    self.name_to_idx[str(row['name']).strip().lower()] = row['pyg_id']
                    
            self.model = MacroscopicDDIGNN(
                in_channels=self.macro_graph_data.num_features,
                hidden_channels=256,
                out_channels=128,
                num_layers=3
            )
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
            self.model.eval()

            self.model_type = ModelType.MACROSCOPIC_GNN
            self.is_loaded = True
            logger.info("Successfully loaded V2 Macroscopic GNN Predictor (~98.6% AUC).")

        except Exception as e:
            logger.error(f"Failed to load Macroscopic GNN Predictor: {e}")
            self.model = None"""

# Split by the bad function and replace
parts = re.split(r'    def _load_macroscopic_gnn\(self\):.*?self\.model = None', text, flags=re.DOTALL)
if len(parts) == 2:
    with open("ddi_api/services/gnn_predictor.py", "w", encoding="utf-8") as f:
        f.write(parts[0] + new_func + parts[1])
    print("Patched!")
else:
    print("Could not match exactly one occurrence.", len(parts))
