import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondType, ChiralType


# =============== SMILES -> graph (adapted from DrugChat) ===============

BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
BOND_DIR = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHI = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: 1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    ChiralType.CHI_OTHER: 3,
}


def bond_dir(bond) -> int:
    return BOND_DIR[bond.GetBondDir()]


def bond_type(bond) -> int:
    return BOND_TYPE[bond.GetBondType()]


def atom_chiral(atom) -> int:
    return CHI[atom.GetChiralTag()]


def atom_to_feature(atom) -> List[int]:
    num = atom.GetAtomicNum() - 1
    if num == -1:
        # wildcard atom '*' (atomic num 0) -> reserve index 118
        num = 118
    return [num, atom_chiral(atom)]


def bond_to_feature(bond) -> List[int]:
    return [bond_type(bond), bond_dir(bond)]


def smiles2graph(smiles_string: str, motif: bool = False, add_lap_pe: bool = True) -> Dict[str, Any]:
    """
    Converts SMILES string to a dict representation:
      - node_feat: np.int64 [N, 2] (atomic_num_minus_1, chiral_tag)
      - edge_index: np.int64 [2, E] (COO)
      - edge_feat: np.int64 [E, 2] (bond_type, bond_dir)
      - lap_pe: torch.float32 [N, 8] (optional)
    """
    if motif:
        # Only works if your environment provides llava.datasets.data_utils.MolGraph
        from llava.datasets.data_utils import MolGraph  # type: ignore

        mol = MolGraph(smiles_string)
        graph = dict()
        graph["edge_index"] = mol.edge_index
        graph["edge_feat"] = mol.edge_attr
        graph["node_feat"] = mol.x
        graph["num_part"] = mol.num_part
        graph["num_nodes"] = len(mol.x)
    else:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles_string}")

        # atoms
        atom_features_list = [atom_to_feature(atom) for atom in mol.GetAtoms()]
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 2
        if len(mol.GetBonds()) > 0:
            edges_list: List[Tuple[int, int]] = []
            edge_features_list: List[List[int]] = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature(bond)
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_attr = np.array(edge_features_list, dtype=np.int64)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)

    if add_lap_pe:
        # requires torch_geometric + scipy
        from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix  # type: ignore

        num_nodes = int(graph["num_nodes"])
        edge_index, _edge_weight = get_laplacian(
            torch.as_tensor(graph["edge_index"], dtype=torch.long),
            normalization="sym",
            num_nodes=num_nodes,
        )
        L = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

        if num_nodes < 100:
            from numpy.linalg import eigh

            eig_vals, eig_vecs = eigh(L.todense())  # type: ignore
        else:
            from scipy.sparse.linalg import eigsh  # type: ignore

            eig_vals, eig_vecs = eigsh(  # type: ignore
                L,
                k=8 + 1,
                which="SA",
                return_eigenvectors=True,
            )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        if len(eig_vals) < 9:
            padding_eig = np.zeros((eig_vecs.shape[0], 9 - len(eig_vals)))
            eig_vecs = np.concatenate((eig_vecs, padding_eig), axis=-1)

        pe = torch.from_numpy(eig_vecs[:, 1 : 8 + 1]).float()  # [N, 8]
        sign = -1 + 2 * torch.randint(0, 2, (8,))
        pe = pe * sign
        graph["lap_pe"] = pe

    return graph


# =============== Graph container expected by encode_mol ===============


@dataclass
class MolData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    lap_pe: Optional[torch.Tensor]
    num_part: List[int]
    num_nodes: int


def graph_dict_to_mol(graph: Dict[str, Any]) -> MolData:
    num_nodes = int(graph["num_nodes"])
    x = torch.as_tensor(graph["node_feat"], dtype=torch.long)
    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
    edge_attr = torch.as_tensor(graph["edge_feat"], dtype=torch.long)
    lap_pe = graph.get("lap_pe", None)
    if lap_pe is not None and not isinstance(lap_pe, torch.Tensor):
        lap_pe = torch.as_tensor(lap_pe, dtype=torch.float32)

    # Default partitioning for non-motif graphs:
    #   num_part[0] = node tokens
    #   num_part[1] = motif tokens (0)
    #   num_part[2] = global tokens (1)  (some encoders expect a pooled token)
    num_part = [num_nodes, 0, 1]
    return MolData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        lap_pe=lap_pe,
        num_part=num_part,
        num_nodes=num_nodes,
    )


# =============== Printing helpers ===============


def summarize_graph(graph: Dict[str, Any], max_show: int = 5) -> Dict[str, Any]:
    edge_index = graph["edge_index"]
    edge_feat = graph["edge_feat"]
    node_feat = graph["node_feat"]
    lap_pe = graph.get("lap_pe", None)

    def shape(x) -> List[int]:
        return list(x.shape)

    summary: Dict[str, Any] = {
        "num_nodes": int(graph["num_nodes"]),
        "node_feat_shape": shape(node_feat),
        "edge_index_shape": shape(edge_index),
        "edge_feat_shape": shape(edge_feat),
        "first_nodes": node_feat[: min(max_show, len(node_feat))].tolist(),
        "first_edges": edge_index[:, : min(max_show, edge_index.shape[1])].tolist()
        if edge_index.shape[1] > 0
        else [],
    }
    if lap_pe is not None:
        summary["lap_pe_shape"] = list(lap_pe.shape)
        summary["lap_pe_first_rows"] = lap_pe[: min(max_show, lap_pe.shape[0])].cpu().tolist()
    else:
        summary["lap_pe_shape"] = None

    return summary


def print_tensor_preview(x: torch.Tensor, rows: int = 3, cols: int = 8) -> None:
    x_cpu = x.detach().cpu()
    r = min(rows, x_cpu.shape[0])
    c = min(cols, x_cpu.shape[1])
    preview = x_cpu[:r, :c]
    print(preview)
    if x_cpu.shape[1] > c:
        print(f"... (showing first {c} / {x_cpu.shape[1]} dims)")


# =============== Encoder selection ===============


def resolve_callable(dotted: str) -> Callable:
    """
    Supports:
      - module:function
      - module.submodule:ClassName (instantiated with no args, then called)
      - module.submodule:obj.attr (callable)
    """
    if ":" not in dotted:
        raise ValueError("Use the format module:callable")
    module_name, sym = dotted.split(":", 1)
    mod = importlib.import_module(module_name)
    obj: Any = mod
    for part in sym.split("."):
        obj = getattr(obj, part)
    if isinstance(obj, type):
        inst = obj()
        if not callable(inst):
            raise TypeError(f"{dotted} resolved to a class, but instance is not callable.")
        return inst
    if not callable(obj):
        raise TypeError(f"{dotted} resolved to a non-callable: {type(obj)}")
    return obj


class DummyGraphEncoder(torch.nn.Module):
    """
    Deterministic, lightweight stand-in that turns (x, lap_pe) into token embeddings.
    This is NOT your trained model; it only exists so the pipeline can be executed
    even when the real LLaVA/graph tower code isn't available yet.
    """

    def __init__(self, hidden_dim: int, use_lap_pe: bool = True, seed: int = 0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)

        self.hidden_dim = hidden_dim
        self.use_lap_pe = use_lap_pe

        in_dim = 2 + (8 if use_lap_pe else 0)
        self.proj = torch.nn.Linear(in_dim, hidden_dim)
        with torch.no_grad():
            # deterministic init
            self.proj.weight.copy_(torch.randn_like(self.proj.weight, generator=g) * 0.02)
            self.proj.bias.zero_()

    def forward(self, mol: MolData, compute_type: torch.dtype = torch.bfloat16) -> torch.Tensor:
        x = mol.x.to(dtype=torch.float32)  # [N, 2]
        if self.use_lap_pe and mol.lap_pe is not None:
            pe = mol.lap_pe.to(dtype=torch.float32)  # [N, 8]
            feats = torch.cat([x, pe], dim=-1)
        else:
            feats = x

        node_tokens = self.proj(feats)  # [N, H]
        # add a simple global token
        global_token = node_tokens.mean(dim=0, keepdim=True)  # [1, H]
        out = torch.cat([node_tokens, global_token], dim=0)  # [N+1, H]
        return out.to(dtype=compute_type)


def load_smiles_from_args(args) -> List[str]:
    smiles: List[str] = []
    if args.smiles:
        smiles.extend(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                smiles.append(line)
    if not smiles:
        raise SystemExit("No SMILES provided. Use --smiles ... or --smiles-file path.txt")
    return smiles


# =============== Main debug ===============


def main():
    parser = argparse.ArgumentParser(
        description="Debug SMILES -> graph -> encoded graph tokens (stop before LLM)."
    )
    parser.add_argument("--smiles", action="append", default=[], help="SMILES string (repeatable)")
    parser.add_argument("--smiles-file", type=str, default=None, help="Text file with one SMILES per line")
    parser.add_argument("--motif", action="store_true", help="Use motif graph builder (requires llava MolGraph)")
    parser.add_argument("--no-lap-pe", action="store_true", help="Disable Laplacian positional encodings")

    parser.add_argument(
        "--encoder",
        type=str,
        default="dummy",
        help="Encoder to use: 'dummy' or 'import:module:callable'.",
    )
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Dummy encoder hidden dim")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Output embedding dtype",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for dummy encoder")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    compute_type = dtype_map[args.dtype]

    smiles_list = load_smiles_from_args(args)

    if args.encoder == "dummy":
        encoder: Callable[[MolData, torch.dtype], torch.Tensor]
        dummy = DummyGraphEncoder(
            hidden_dim=args.hidden_dim,
            use_lap_pe=not args.no_lap_pe,
            seed=args.seed,
        )

        def encoder_fn(mol: MolData, compute_type: torch.dtype = compute_type) -> torch.Tensor:
            return dummy(mol, compute_type=compute_type)

    elif args.encoder.startswith("import:"):
        dotted = args.encoder[len("import:") :]
        resolved = resolve_callable(dotted)

        def encoder_fn(mol: MolData, compute_type: torch.dtype = compute_type) -> torch.Tensor:
            out = resolved(mol, compute_type=compute_type)  # type: ignore
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"Imported encoder returned {type(out)}, expected torch.Tensor")
            return out.to(dtype=compute_type)

    else:
        raise SystemExit("Unsupported --encoder. Use 'dummy' or 'import:module:callable'.")

    for i, smi in enumerate(smiles_list):
        print("=" * 80)
        print(f"[{i}] Raw SMILES:\n{smi}")

        rd_mol = Chem.MolFromSmiles(smi)
        if rd_mol is None:
            print(f"[{i}] ERROR: RDKit cannot parse this SMILES.")
            continue

        graph = smiles2graph(smi, motif=args.motif, add_lap_pe=not args.no_lap_pe)
        summary = summarize_graph(graph)
        print(f"[{i}] Graph (after smiles2graph) summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        mol = graph_dict_to_mol(graph)

        with torch.no_grad():
            feats = encoder_fn(mol, compute_type=compute_type)

        print(f"[{i}] Encoded graph features (ready for LLM, not injected):")
        print(f"  dtype: {feats.dtype}")
        print(f"  shape: {list(feats.shape)}  # [num_graph_tokens, hidden_dim]")
        print("  preview (first rows, first dims):")
        print_tensor_preview(feats, rows=3, cols=8)


if __name__ == "__main__":
    main()

