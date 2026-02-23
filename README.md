### SMART-HIGHT debug scripts

This repo currently contains a **single runnable debug pipeline** that stops right before the LLM:

- `debug_smiles_to_graph_tokens.py`: **SMILES → graph → encoded graph-token embeddings** (ready to be injected into an LLM, but not injected here).

There is also an earlier helper (`smiles_graph_debug.py`), but `debug_smiles_to_graph_tokens.py` is the one that is self-contained and verified runnable in your environment.

---

### What this script is for

When building a multimodal LLM that accepts molecules as graphs (similar to how LLaVA accepts images), there are 3 important intermediate representations to inspect:

- **Raw input**: a SMILES string
- **Graph form**: nodes + edges + features (and optional Laplacian positional encodings)
- **LLM-ready embeddings**: a float tensor shaped like \([num\_graph\_tokens, hidden\_dim]\) that can be inserted where an `<image>` token would be

`debug_smiles_to_graph_tokens.py` prints all three.

---

### Requirements (your working venv)

In your `.venv-312`, the script needs:

- **Required**:
  - `torch`
  - `rdkit` (RDKit)
  - `numpy`
- **If Laplacian PE is enabled (default)**:
  - `torch-geometric`
  - `scipy`

If you want to avoid `torch-geometric` + `scipy`, run with `--no-lap-pe`.

---

### How to run

From `C:\Users\andre\SMART-HIGHT`:

```powershell
.\.venv-312\Scripts\python.exe .\debug_smiles_to_graph_tokens.py --smiles "C=CC1CN2CCC1CC2C(O)c1ccnc2ccc(OC)cc12"
```

Disable Laplacian PE:

```powershell
.\.venv-312\Scripts\python.exe .\debug_smiles_to_graph_tokens.py --smiles "CCO" --no-lap-pe
```

Read SMILES from a file (one per line):

```powershell
.\.venv-312\Scripts\python.exe .\debug_smiles_to_graph_tokens.py --smiles-file .\smiles.txt
```

Control the dummy encoder output shape:

```powershell
.\.venv-312\Scripts\python.exe .\debug_smiles_to_graph_tokens.py --smiles "CCO" --hidden-dim 512 --dtype float32
```

---

### Code walkthrough (file: `debug_smiles_to_graph_tokens.py`)

#### 1) SMILES → graph (`smiles2graph`)

The function `smiles2graph(...)` converts a SMILES string into a Python dict with:

- **`node_feat`**: `np.int64` array \([N, 2]\)
  - column 0: `atomic_number_minus_1` (C=6 → 5, O=8 → 7, etc.)
  - column 1: RDKit chiral tag encoded as a small int
- **`edge_index`**: `np.int64` array \([2, E]\)
  - COO format listing directed edges (each bond is added in both directions)
- **`edge_feat`**: `np.int64` array \([E, 2]\)
  - column 0: bond type (single/double/triple/aromatic)
  - column 1: bond direction
- **`num_nodes`**: integer \(N\)
- **`lap_pe`** (optional): `torch.float32` tensor \([N, 8]\)
  - Laplacian positional encoding (8 eigenvectors)

This is the “graph form” you want to inspect before encoding.

#### 2) Graph dict → model input object (`graph_dict_to_mol`)

Your real `encode_mol` expects an object with attributes (not a dict). The script wraps the dict into a `MolData` object containing:

- `x`: node features as a tensor
- `edge_index`: tensor \([2, E]\)
- `edge_attr`: tensor \([E, 2]\)
- `lap_pe`: tensor \([N, 8]\) (optional)
- `num_nodes`: integer
- `num_part`: a partition list for node/motif/global tokens

For non-motif graphs, this script sets:

- `num_part = [num_nodes, 0, 1]`

Meaning:

- `num_part[0]`: node tokens = all atoms
- `num_part[1]`: motif tokens = 0 (none)
- `num_part[2]`: global tokens = 1 (a pooled token; many encoders add one)

#### 3) Graph → “LLM-ready” tokens (encoder)

The script supports two encoder modes:

- **Default**: `--encoder dummy`
  - Uses `DummyGraphEncoder` to produce a deterministic tensor shaped like a real projector output.
  - This is **not** your trained model; it’s only to verify the pipeline end-to-end.
- **Real model (later)**: `--encoder import:module:callable`
  - You can point it at a real `encode_mol(mol, compute_type=...) -> Tensor` function once your model code is importable in this repo/environment.

In both cases, the expected output is:

- **`graph_features`**: `torch.Tensor` shaped \([num_graph_tokens, hidden_dim]\)

This is the representation that is compatible with an LLM’s hidden size and can replace the `<image>` placeholder embedding sequence (in your multimodal input builder).

---

### Output walkthrough (what each printed block means)

The script prints one molecule at a time.

#### A) Raw SMILES

Example:

- `Raw SMILES:` followed by the literal input string.

If RDKit cannot parse it, you will see:

- `ERROR: RDKit cannot parse this SMILES.`

#### B) Graph summary (after `smiles2graph`)

Example fields:

- **`num_nodes`**
  - Number of atoms in the molecule.
- **`node_feat_shape: [N, 2]`**
  - Two integer features per atom: \([atomic_number_minus_1, chiral_tag]\).
- **`edge_index_shape: [2, E]`**
  - A directed COO edge list. Since each bond is added both ways, \(E\) is typically \(2 ×\) number_of_bonds.
- **`edge_feat_shape: [E, 2]`**
  - Two integer features per directed edge: \([bond_type, bond_dir]\).
- **`first_nodes`**
  - The first few rows of `node_feat` so you can sanity check atoms/chirality quickly.
- **`first_edges`**
  - The first few directed edges from `edge_index`.
- **`lap_pe_shape`**
  - `None` if you ran with `--no-lap-pe`.
  - Otherwise `[N, 8]` for 8 Laplacian PE dimensions per node.
- **`lap_pe_first_rows`**
  - First few LapPE rows (floats). This helps confirm the PE exists and has the right shape.

#### C) Encoded graph features (ready for LLM, not injected)

Example fields:

- **`dtype`**
  - The output dtype you selected (default `bfloat16`).
- **`shape: [num_graph_tokens, hidden_dim]`**
  - The exact “LLM-ready” layout.
  - With the dummy encoder, `num_graph_tokens = num_nodes + 1` because it adds a simple global token.
- **`preview`**
  - Prints the first few tokens and the first few dimensions so you can confirm:
    - values are finite
    - dimensions match expectation
    - output changes when LapPE is enabled/disabled

This is the final representation you want before calling any LLM code.

---

### Troubleshooting

- **`ModuleNotFoundError: No module named 'rdkit'`**
  - Install RDKit into the active venv:
    - `pip install rdkit`
- **LapPE enabled errors mentioning `scipy`**
  - Install SciPy:
    - `pip install scipy`
- **You only want the pipeline without LapPE**
  - Run with `--no-lap-pe`

