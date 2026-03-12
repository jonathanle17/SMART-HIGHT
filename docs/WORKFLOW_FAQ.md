# Workflow FAQ

## 1. Run on 200 samples is still running – wait or kill?

- **With fingerprint caching**, 200 samples should usually finish in a few minutes (depending on retrieval size). If it’s been **more than ~5–10 minutes**, it’s reasonable to **stop the run** (Ctrl+C).
- For quicker checks:
  - **Single random molecule:**  
    `python run_pipeline_with_dataset.py --single-random`
  - **Fewer samples:**  
    `python run_pipeline_with_dataset.py --max-samples 50`
- If the retrieval list is very large (e.g. 100k+), the first run builds the fingerprint cache once; that can take a few minutes before the 200-sample loop.

---

## 2. Where are the raw spectrum files? (SMART-Moonshot)

In **SMART-Moonshot**, spectra are **not** one file per sample. They are stored in **Arrow/Parquet** tables under the dataset root:

```
{dataset_root}/arrow/{split}/{MODALITY}.parquet
```

- **Modalities:**  
  `MassSpec.parquet`, `HSQC_NMR.parquet`, `H_NMR.parquet`, `C_NMR.parquet`
- **Splits:**  
  `train`, `val`, `test`

So you get:

- **MassSpec (MS):**  
  `.../arrow/train/MassSpec.parquet`, `.../arrow/val/MassSpec.parquet`, etc.
- **NMR:**  
  `.../arrow/train/HSQC_NMR.parquet`, `.../arrow/train/H_NMR.parquet`, `.../arrow/train/C_NMR.parquet`, and the same under `val/` and `test/`.

**How SMART-Moonshot loads them:**

- **File:** `src/modules/data/inputs.py`
- **Classes:**  
  `SpectralInputLoader`, `MARINAInputLoader`, `SPECTREInputLoader`
- **Usage:**  
  They use `_get_tensor(idx, 'MassSpec')` (or `'HSQC_NMR'`, etc.), which reads from `ArrowTensorStore(path)` for the parquet at  
  `os.path.join(root, "arrow", split, f"{mod}.parquet")`.

So:

- **“Raw spectrum”** = rows in those `.parquet` files, indexed by sample `idx`.
- **Where to look on disk:**  
  Unzip MoonshotDatasetv3 and check for an `arrow/` folder; the paths above are relative to that dataset root. If your zip only has `index.pkl` and `retrieval.pkl`, the `arrow/` shards may be in a different download or a different layout (you’d need to confirm with the dataset provider).

---

## 3. Incorporating the HIGHT encoder into this pipeline

**Can it be done?** Yes, but it’s not a one-line swap.

**Why:**

- **HIGHT** (LFhase/HIGHT) uses:
  - Its own graph format: **`MolGraph(smiles)`** in `llava.datasets.data_utils` (atom features = atomic number + degree; edges = bond type + in-ring; **motif nodes** and a **super node**; `num_part = (num_atoms, num_motif, 1)`).
  - A **graph tower** built with `build_graph_tower` (e.g. `vqvae2` or `hvqvae2`) and an **mm_projector**.
  - **`encode_mol(mol, compute_type=...)`** on the full LLaVA model, which calls the graph tower then the projector.
- **Our pipeline** uses:
  - **MolData** with `x`, `edge_index`, `edge_attr`, `lap_pe`, `num_part` from `debug_smiles_to_graph_tokens` (no motif layer, `num_part = [num_nodes, 0, 1]`).

So the **graph format and model** are different. To use the real HIGHT encoder here you’d need to:

1. **Install HIGHT** (clone repo, install deps, e.g. `pip install -e .` and their `requirements.txt`).
2. **Download their checkpoints** (graph tower + optional mm_projector), e.g. from Hugging Face.
3. **Build the model** (e.g. from `llava.model.builder` / `build_graph_tower`) and load weights.
4. **Feed HIGHT-style graphs:**
   - Either call **`MolGraph(smiles)`** from HIGHT’s `llava.datasets.data_utils` (so you get their motif/super-node structure), then pass that object into their **`encode_mol`**.
   - Or implement an **adapter** that converts our `MolData` → their graph (add motif/super node and match feature conventions); that’s more work and easy to get wrong.

**Practical “easy” path:**

- Add a **small wrapper** in this repo that:
  - Imports HIGHT’s `MolGraph` and their model’s `encode_mol` **only when HIGHT is installed**.
  - For each SMILES: `mol = MolGraph(smiles)`, then call `encode_mol(mol, compute_type=...)` and use the returned tensor as the graph embedding.
- Keep our **DummyGraphEncoder** as the default when HIGHT isn’t installed or checkpoint isn’t available.

So: **yes, you can incorporate the HIGHT encoder**, but it requires adding HIGHT as a dependency, loading their checkpoints, and using their graph format (e.g. via their `MolGraph`). The repo includes an optional **HIGHT encoder wrapper** (see `hight_encoder_wrapper.py`) that you can enable once HIGHT is installed and checkpoint path is set.
