import torch
from typing import Callable, Dict, Any, List


class MolData:
    """
    Minimal stand-in for the graph object that `encode_mol` expects.
    It just carries attributes as plain tensors: x, edge_index,
    edge_attr, lap_pe, num_part, num_nodes.
    """

    def __init__(self):
        self.x = None
        self.edge_index = None
        self.edge_attr = None
        self.lap_pe = None
        self.num_part = None
        self.num_nodes = None


def graph_dict_to_mol(graph_dict: Dict[str, Any]) -> MolData:
    """
    Adapter from the plain dict returned by `smiles2graph`
    to a simple object with the attributes that `encode_mol`
    expects (`x`, `edge_index`, `edge_attr`, `lap_pe`, `num_part`).
    """
    num_nodes = graph_dict["num_nodes"]
    x = torch.as_tensor(graph_dict["node_feat"])
    edge_index = torch.as_tensor(graph_dict["edge_index"])
    edge_attr = torch.as_tensor(graph_dict["edge_feat"])
    lap_pe = graph_dict["lap_pe"]

    mol = MolData()
    mol.x = x
    mol.edge_index = edge_index
    mol.edge_attr = edge_attr
    mol.lap_pe = lap_pe
    mol.num_nodes = num_nodes

    # Minimal convention:
    #   num_part[0]: number of node tokens
    #   num_part[1]: number of motif tokens (0 here)
    #   num_part[2]: number of global tokens (1 here, e.g. pooled token)
    mol.num_part = [num_nodes, 0, 1]

    return mol


def describe_graph_dict(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a lightweight, print-friendly summary of the graph dict.
    """
    edge_index = graph_dict["edge_index"]
    edge_feat = graph_dict["edge_feat"]
    node_feat = graph_dict["node_feat"]
    lap_pe = graph_dict["lap_pe"]

    summary = {
        "num_nodes": int(graph_dict["num_nodes"]),
        "node_feat_shape": list(node_feat.shape),
        "edge_index_shape": list(edge_index.shape),
        "edge_feat_shape": list(edge_feat.shape),
        "lap_pe_shape": list(lap_pe.shape),
        "first_nodes": node_feat[: min(5, len(node_feat))].tolist(),
        "first_edges": edge_index[:, : min(5, edge_index.shape[1])].tolist()
        if edge_index.shape[1] > 0
        else [],
    }
    return summary


def debug_smiles_pipeline(
    smiles_list: List[str],
    smiles2graph_fn: Callable[[str], Dict[str, Any]],
    encode_mol_fn: Callable[[Any], torch.Tensor],
    compute_type: torch.dtype = torch.bfloat16,
) -> None:
    """
    For each SMILES string, print:
      1) The raw SMILES.
      2) A summary of the graph representation from `smiles2graph_fn`.
      3) The shape (and first few rows) of the encoded graph features
         returned by `encode_mol_fn`, ready for LLM injection.

    This intentionally stops right before the LLM: no text tokenization,
    no injection into the language model, just the graph side.
    """
    from rdkit import Chem

    for idx, smi in enumerate(smiles_list):
        print("=" * 80)
        print(f"[{idx}] Raw SMILES:")
        print(smi)

        # 1) Validate SMILES
        mol_rdkit = Chem.MolFromSmiles(smi)
        if mol_rdkit is None:
            print(f"[{idx}] ERROR: invalid SMILES, cannot be parsed by RDKit.")
            continue

        # 2) SMILES -> graph dict
        graph_dict = smiles2graph_fn(smi)
        graph_summary = describe_graph_dict(graph_dict)

        print(f"[{idx}] Graph summary after `smiles2graph`:")
        for k, v in graph_summary.items():
            print(f"  {k}: {v}")

        # 3) Graph dict -> mol object compatible with encode_mol
        mol_graph = graph_dict_to_mol(graph_dict)

        # 4) Encode to graph tokens ready for LLM
        with torch.no_grad():
            graph_features = encode_mol_fn(mol_graph)
            # Ensure desired compute type if caller expects that
            graph_features = graph_features.to(dtype=compute_type)

        print(f"[{idx}] Encoded graph features (ready for LLM):")
        print(f"  shape: {list(graph_features.shape)}  # [num_graph_tokens, hidden_dim]")

        # Show the first few graph tokens for inspection
        num_show = min(3, graph_features.shape[0])
        print(f"  first {num_show} token embeddings (truncated):")
        print(graph_features[:num_show].cpu())


if __name__ == "__main__":
    """
    Example usage (pseudo-code):

        from your_module.smiles2graph import smiles2graph
        from your_model_wrapper import ModelWrapper

        model_wrapper = ModelWrapper(...)

        def encode_mol_wrapper(mol):
            return model_wrapper.encode_mol(mol)  # must return [N, hidden_dim]

        smiles_list = [
            "COC(C=C(C)C)CC(C)C1CCC2(C)C3C=CC45OCC3(CCC12C)C4CCC(OC1OC(CO)C(O)C(O)C1O)C5(C)C",
            "C=CC1CN2CCC1CC2C(O)c1ccnc2ccc(OC)cc12",
        ]

        debug_smiles_pipeline(smiles_list, smiles2graph, encode_mol_wrapper)

    Adjust the imports above to match your codebase.
    """
    print(
        "This module defines `debug_smiles_pipeline`. "
        "Import it and call `debug_smiles_pipeline(...)` with your own "
        "`smiles2graph` and `encode_mol` implementations."
    )

