import os
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: torch_geometric is not installed. Will just save plain tensors.")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit is not installed. SMILES will just be saved as strings, not fingerprints.")

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', os.getenv('NEO4J_USERNAME'))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def generator_morgan_fingerprint(smiles, radius=2, bits=1024):
    """Convert SMILES to a Morgan Fingerprint array using RDKit."""
    if not RDKIT_AVAILABLE or not smiles:
        return np.zeros(bits)
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros(bits)
        
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    print("Connecting to Neo4j to extract Heterogeneous Graph Data...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    nodes_data = []
    edges_data = []

    with driver.session() as session:
        # 1. Fetch Valid Nodes (Drugs with SMILES)
        print("Extracting nodes...")
        query_nodes = "MATCH (d:Drug) WHERE d.smiles IS NOT NULL RETURN id(d) as nid, d.name as name, d.smiles as smiles, d.therapeutic_class as t_class"
        res_nodes = session.run(query_nodes)
        
        for record in res_nodes:
            nodes_data.append({
                "nid": record["nid"],
                "name": record["name"],
                "smiles": record["smiles"],
                "t_class": record["t_class"] or "Unknown"
            })
            
        # 2. Fetch Valid Edges (Only edges where both ends are in our valid nodes list)
        print("Extracting edges...")
        query_edges = """
        MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
        WHERE d1.smiles IS NOT NULL AND d2.smiles IS NOT NULL
        RETURN id(d1) as src, id(d2) as dst, r.severity as severity
        """
        res_edges = session.run(query_edges)
        
        for record in res_edges:
            edges_data.append({
                "src": record["src"],
                "dst": record["dst"],
                "severity": record["severity"] or "Unknown"
            })

    driver.close()
    
    df_nodes = pd.DataFrame(nodes_data)
    df_edges = pd.DataFrame(edges_data)
    
    print(f"Extracted {len(df_nodes)} nodes and {len(df_edges)} valid edges.")
    
    if len(df_nodes) == 0:
        print("No valid nodes found. Exiting.")
        return

    # --- Data Processing for PyG ---
    print("Processing node indices and features...")
    
    # Map Neo4j DB IDs to continuous 0-indexed PyG indices
    id_mapping = {nid: i for i, nid in enumerate(df_nodes['nid'])}
    
    # 1. Generate Chemical Embeddings (Fingerprints)
    if RDKIT_AVAILABLE:
        print("Generating RDKit Morgan Fingerprints...")
        node_features = np.array([generator_morgan_fingerprint(s) for s in df_nodes['smiles']])
    else:
        # Dummy features if RDKit is missing
        node_features = np.zeros((len(df_nodes), 1024))
        
    # 2. Generate Biological Class Embeddings (One-Hot Encoding)
    print("One-hot encoding therapeutic classes...")
    t_classes = pd.get_dummies(df_nodes['t_class']).values
    
    # Combine Chemical + Biological features into one Tensor per node
    combined_features = np.concatenate([node_features, t_classes], axis=1)
    x_tensor = torch.tensor(combined_features, dtype=torch.float)
    
    # 3. Process Edges
    src_indices = [id_mapping[src] for src in df_edges['src']]
    dst_indices = [id_mapping[dst] for dst in df_edges['dst']]
    
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    
    print("Graph Structure Built:")
    print(f" - Node Feature Tensor Shape: {x_tensor.shape} (Chemical Bits + Biological Classes)")
    print(f" - Edge Index Tensor Shape: {edge_index.shape}")
    
    # Save the PyG Data Object
    save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_dir, exist_ok=True)
    
    if PYG_AVAILABLE:
        graph_data = Data(x=x_tensor, edge_index=edge_index)
        torch.save(graph_data, os.path.join(save_dir, "neo4j_gnn_dataset.pt"))
    else:
        torch.save({"x": x_tensor, "edge_index": edge_index}, os.path.join(save_dir, "neo4j_gnn_raw_tensors.pt"))
        
    # Save mapping for reference (so we map PyG results back to Drug names in the app)
    df_nodes[['name', 'smiles', 't_class']].to_csv(os.path.join(save_dir, "node_mapping.csv"), index_label="pyg_id")
    
    print(f"Success! Dataset saved to {save_dir}/")

if __name__ == '__main__':
    main()
